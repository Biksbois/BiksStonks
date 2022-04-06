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
