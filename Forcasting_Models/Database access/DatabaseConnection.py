import psycopg2 as pg
### Connect to the database
class DatabaseConnection:
    def connect():
        try:
            conn = pg.connect(
                "dbname='stonksdb' user='postgres' host='localhost' password='stonk'")
            return conn
        except:
            print("I am unable to connect to the database")
            return None
test = DatabaseConnection.connect()
print(test)