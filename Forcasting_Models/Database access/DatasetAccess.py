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
            result.append(self.conn.query("SELECT "+self.convertListToString(column)+" FROM stock WHERE identifier = '" + str(company[0]) + "'"))
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
    
def PlotCloseValue(indexes=slice(1)):
    import matplotlib.pyplot as plt
    plt.plot(GetCloseValue(indexes))
    plt.show()
print(GetCloseValue()[0])
# [(Decimal('3.39'),), (Decimal('3.13'),), (Decimal('3.38'),),)]
PlotCloseValue()