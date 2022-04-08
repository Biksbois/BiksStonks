import utils.secrets as settings
import psycopg2 as pg
import pandas as pd
from objects.dataset import SentimentDataset
from tqdm import tqdm
from tqdm import trange


class DatabaseAccess:
    def __init__(self):
        self.datasets = {}

    def get_rows(self, query):
        with self._connect() as conn:
            data = pd.read_sql_query(query, conn)

            return data

    def dataframe_to_db(
        self, df, function, dataset_function, translator, url, source, description
    ):
        succs = []
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                # for _, row in df.iterrows():
                for i in tqdm(range(len(df))):
                    row = df.iloc[i]

                    datasetid = self._get_dataset_id(
                        translator, row, source, url, cur, description, dataset_function
                    )
                    succ = self._insert_article(datasetid, row, cur, function)
                    succs.append(succ)
                print(len(succs) == len(df))
                cur.close()
                print(f"dataset and rows successfully inserted")

        except Exception as e:
            raise Exception(str(e))

    def _insert_article(self, datasetid, row, cur, function):
        return cur.callproc(
            function,
            (
                datasetid,
                row["release_date"],
                row["source_headline"],
                row["target_headline"],
                row["neg"],
                row["pos"],
                row["neu"],
                row["compound"],
                row["url"],
                row["companies"],
            ),
        )

    def _get_dataset_id(
        self, translator, row, source, url, cur, description, dataset_function
    ):
        dataset = SentimentDataset(
            translator,
            row["source_language"],
            row["target_language"],
            source,
            url,
            row["category"],
        )

        if dataset in self.datasets:
            datasetid = self.datasets[dataset]
        else:
            cur.callproc(
                dataset_function,
                (
                    translator,
                    row["source_language"],
                    row["target_language"],
                    source,
                    url,
                    description,
                    row["category"],
                ),
            )
            datasetid = cur.fetchone()[0]

            self.datasets[dataset] = datasetid
        return datasetid

    def _connect(self):
        try:
            conn = pg.connect(
                database=settings.get_database(),
                user=settings.get_user(),
                password=settings.get_pasword(),
            )
            print(f"Connection to '{settings.get_database()}' made successfully")
            return conn
        except Exception as e:
            raise Exception(
                f"Unable to connection to '{settings.get_database()}'. {str(e)}"
            )
