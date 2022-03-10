from DatabaseConnection import DatabaseConnection
class DatasetAccess:
    def __init__(self):
        self.conn = DatabaseConnection()
    
    def getAllCompanyes(self):
        AllCompanies = self.conn.query("SELECT * FROM dataset")
        return AllCompanies
    
    def getStockFrom(self, StockSymbol):
        company = self.conn.query("SELECT * FROM dataset WHERE symbol = '" + StockSymbol + "'")
DatasetAccess = DatasetAccess()
companys = DatasetAccess.getAllCompanyes()
    
print("selecting the first stock")
print(companys[0].symbol)