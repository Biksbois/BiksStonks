import psycopg2 as pg
import utils.settings_utils as settings

### Connect to the database
class DatabaseConnection:
    def __init__(self):
        self.conn = self.connect()

    def connect(self):
        try:
            self.conn = pg.connect(
                database=settings.get_database(),
                user=settings.get_user(),
                password=settings.get_pasword(),
            )
            print("Connection made succ")
            return self.conn
        except Exception as e:
            print("I am unable to connect to the database")
            print(str(e))
            return None

    def close(self):
        self.conn.close()

    def GetConnector(self):
        return self.conn

    ### send query to database
    def query(self, query):
        cur = self.conn.cursor()
        cur.execute(query)
        return cur.fetchall()


# Database = DatabaseConnection()
# stock = Database.query("SELECT * FROM stock")
# for s in stock:
#     print(s)
