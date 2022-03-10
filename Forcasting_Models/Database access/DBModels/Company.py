from locale import currency
from unicodedata import category


class Company:
    identifier = ''
    assetType = ''
    currencycode = ''
    description = ''
    exchange = ''
    groupid = ''
    issuercountry = ''
    primarylisting = ''
    summatype = ''
    symbol = ''
    category = ''
    def __init__(self, identifier, assetType, currencycode, description, exchange, groupid, issuercountry, primarylisting, summatype, symbol, category):
        self.identifier = identifier
        self.assetType = assetType
        self.currencycode = currencycode
        self.description = description
        self.exchange = exchange
        self.groupid = groupid
        self.issuercountry = issuercountry
        self.primarylisting = primarylisting
        self.summatype = summatype
        self.symbol = symbol
        self.category = category
    