from utils.db_access import DatabaseAccess
import re


class CompaniesInHeadline:
    def __init__(self):
        self.company_names = self._get_companies()

    def get_companies_in_headlinen(self, headline):
        companines_in_headline = []
        for company in self.company_names:
            for c in company[1:]:
                if re.search(r"\b" + c.lower() + r"\b", headline.lower()):
                    companines_in_headline.append(company[0])
                    break
        return companines_in_headline

    def _get_companies(self):
        db = DatabaseAccess()
        datasets = db.get_rows("SELECT identifier, description FROM dataset")
        ds = [
            [datasets.iloc[i].identifier, datasets.iloc[i].description]
            for i in range(len(datasets["description"]))
        ]

        for d in ds:
            if " A/S" in d[1]:
                d[1] = d[1].replace(" A/S", "")
            if d[1][-2:] == " B":
                d[1] = d[1][:-2]
            if d[1][-2:] == " A":
                d[1] = d[1][:-2]
            if d[1] in self._costume_names():
                d.extend(self._costume_names()[d[1]])

        return ds

    def _costume_names(self):
        return {
            "Netcompany Group": ["Netcompany"],
            "Nordea Bank Abp": ["Nordea Bank"],
            "Atlantic Petroleum P/F": ["Atlantic Petroleum"],
            "Bang & Olufsen Holding": ["Bang & Olufsen", "Bang og Olufsen", "B&O"],
            "Flugger Group": ["Flügger"],
            "Rockwool International": ["Rockwool"],
            "SAS AB": ["SAS"],
            "Spar Nord Bank": ["Spar Nord"],
            "TORM Plc": ["TORM"],
            "Vestas Wind Systems": ["Vestas"],
            "BankNordik P/F": ["BankNordik"],
            "Boozt AB": ["Boozt"],
            "Aalborg Boldspilklub": ["aab"],
            "Harboes Bryggeri": ["Harboe"],
            "Københavns Lufthavne": ["Københavns Lufthavn"],
            "Lån og Spar Bank": ["Lån og Spar"],
            "Onxeo SA (Copenhagen)": ["Onxeo SA"],
            "Silkeborg IF Invest": ["Silkeborg IF", "SIF"],
            "A.P. Møller - Mærsk": ["Mærsk", "A.P. Møller", "Maersk"],
            "Trifork Holding": ["Trifork", "Trifork Holding"],
            "Brøndbyernes IF": ["Brøndby IF", "BIF"],
            "PARKEN Sport & Entertainment": ["parken"],
        }
