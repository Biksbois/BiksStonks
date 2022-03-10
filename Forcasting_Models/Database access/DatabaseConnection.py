import psycopg2 as pg
### Connect to the database
class DatabaseConnection:
    def connect():
        try:
            conn = pg.connect(
                "dbname='stonksdb' user='postgres' host='localhost' password='stonk'")
            print("Connection made succ")
            return conn
        except:
            print("I am unable to connect to the database")
            return None

    def query(conn, query):
        cur=conn.cursor()
        cur.execute(query)
        return cur.fetchall()

test = DatabaseConnection.connect()
for stock in DatabaseConnection.query(test,"SELECT * FROM stock"):
    print(stock)