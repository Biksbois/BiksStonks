from DatabaseConnection import DatabaseConnection
class DatasetAccess:
    def __init__(self):
        self.conn = DatabaseConnection()
    
    def getAllCompanyes(self):
        AllCompanies = self.conn.query("SELECT * FROM dataset")
        return AllCompanies
    
    def getStockFrom(self, stockName):
        print(test)
DatasetAccess = DatasetAccess()
for data in DatasetAccess.getAllCompanyes():
    print(data)