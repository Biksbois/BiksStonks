from unittest import result
from DatabaseConnection import DatabaseConnection
class DatasetAccess:
    def __init__(self):
        self.conn = DatabaseConnection()
    
    def getAllcompanies(self):
        AllCompanies = self.conn.query("SELECT * FROM dataset")
        return AllCompanies
    
    def getStockFromSymbol(self, StockSymbol, column = '*'):
        company = self.conn.query("SELECT * FROM dataset WHERE symbol = '" + StockSymbol + "'")
        self.getStockFromCompany(company, column)
        return company
    
    def getStockFromCompany(self, companies, column = '*'):
        result = []
        for company in companies:
            result.append(self.conn.query("SELECT '"+self.convertListToString(column)+"' FROM stock WHERE identifier = '" + str(company[0]) + "'"))
            print("SELECT '"+self.convertListToString(column)+"' FROM stock WHERE identifier = '" + str(company[0]) + "'")
        return result
    
    def convertListToString(self, column):
        if type(column) != list:
            return column
        result = ''
        for item in column:
            result += item + ', '
        return result[:-2]
    
def GetCloseValue(indexes=slice(1)):
    dbAccess = DatasetAccess()
    return dbAccess.getStockFromCompany(dbAccess.getAllcompanies()[indexes], 'close')
    
test = GetCloseValue()
print("start")
# for t in test:
#     print(t)