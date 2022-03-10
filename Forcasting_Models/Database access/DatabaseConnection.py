import psycopg2 as pg
### Connect to the database
class DatabaseConnection:
    def __init__(self):
        self.conn = self.connect()
    def connect(self):
        try:
            self.conn = pg.connect(
                "dbname='stonksdb' user='postgres' host='localhost' password='stonk'")
            print("Connection made succ")
            return self.conn
        except:
            print("I am unable to connect to the database")
            return None
    def close(self):
        self.conn.close()
        
    ### send query to database
    def query(self, query):
        cur = self.conn.cursor(cursor_factory=pg.extras.NamedTupleCursor)
        cur.execute(query)
        return cur.fetchall()
# Database = DatabaseConnection()
# stock = Database.query("SELECT * FROM stock")
# for s in stock:
#     print(s) 
