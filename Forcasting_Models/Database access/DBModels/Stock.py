from DBModels.PricePoint import PricePoint
from DBModels.PricePoint import Company


class stock:
    PricePoints = []
    Company = ""
    def __init__(self, PricePoints, Company):
        self.PricePoints = PricePoints
        self.Company = Company