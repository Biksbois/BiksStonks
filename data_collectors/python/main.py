from unittest.mock import sentinel
from utils.sentiment_analysis import SentimentAnalysis
from utils.companines import CompaniesInHeadline
from utils.translator import Translator
from utils.db_access import DatabaseAccess
from sentiment_sources.boersen import Boersen
from utils.logger import OwnLogger
import logging
import time

if __name__ == "__main__":
    name = "datacollector"

    print("starting...")
    db = DatabaseAccess()
    analyzer = SentimentAnalysis()
    danish_translator = Translator(source_language="da", target_language="en")
    company_in_headline = CompaniesInHeadline()

    boersen = Boersen(analyzer, danish_translator, company_in_headline, db)

    news_source = [boersen]

    for source in news_source:
        try:
            print(f"About to fetch data from '{source.name}'")
            source.start()
        except Exception as e:
            print(f"ERROR: {str(e)}")

    print("ending...")
    time.sleep(2)
