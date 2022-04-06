import utils.secrets as settings
import psycopg2 as pg
import pandas as pd


class DatabaseAccess:
    def connect(self):
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

    def get_rows(self, query):
        with self.connect() as conn:
            data = pd.read_sql_query(query, conn)

            return data

    def dataframe_to_db(self, df, function):
        try:
            with self.connect() as conn:
                cur = conn.cursor()
                for index, row in df.iterrows():
                    a = cur.callproc(
                        function,
                        (
                            row["release_date"],
                            row["source_headline"],
                            row["target_headline"],
                            row["source_language"],
                            row["target_language"],
                            row["neg"],
                            row["pos"],
                            row["neu"],
                            row["compound"],
                            row["url"],
                            row["companies"],
                            row["category"],
                        ),
                    )
                    print(a)
                cur.close()

        except Exception as e:
            print(str(e))
