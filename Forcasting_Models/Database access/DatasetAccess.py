from unittest import result
from DatabaseConnection import DatabaseConnection
class DatasetAccess:
    def __init__(self):
        self.conn = DatabaseConnection()
    
    def getAllcompanies(self):
        AllCompanies = self.conn.query("SELECT * FROM dataset")
        return AllCompanies
    
    def getStockFromSymbol(self, StockSymbol):
        company = self.conn.query("SELECT * FROM dataset WHERE symbol = '" + StockSymbol + "'")
        return company
    
    def getStockFromCompany(self, companies):
        result = []
        for company in companies:
            result.append(self.conn.query("SELECT * FROM stock WHERE identifier = '" + company[0] + "'"))
        return company
    
DatasetAccess = DatasetAccess()
companies = DatasetAccess.getAllcompanies()
    
print("selecting the first stock")
print(companies[0][9])

company = DatasetAccess.getStockFromCompany(companies[slice(0,2)])
print(company)
