from cmath import log
from utils.sentiment_analysis import SentimentAnalysis
from utils.translator import Translator
from utils.scraper import call_url_get_bs4
from bs4 import BeautifulSoup
import logging


class Boersen:
    def __init__(self, analyzer: SentimentAnalysis, translator: Translator, logger):
        self.analyzer = analyzer
        self.translator = translator
        self.logger = logger
        self.name = "boersen"
        self.base_url = "https://api.borsen.dk/nyheder/side"

    def start(self):
        headlines = self._get_headlines()

        headlines = self._assing_scores_to_headlines(headlines)

        self._insert_headlines_to_db(headlines)

    def _get_headlines(self):
        self.logger.info("{source} is about to get headlinens", source=self.name)

        i = 1
        while True:
            soup = call_url_get_bs4(f"{self.base_url}/{i}")

            if self._no_more_pages(soup):
                break
            else:
                headlines = self._get_headlines_from_soup(soup)
            i += 1

        headlines = []

        self.logger.info("{amount} headlines fetched", amount=len(headlines))

        return headlines

    def _no_more_pages(soup):
        soup.find

    def _assing_scores_to_headlines(self, headlines):
        self.logger.info(
            "scores are about to be assigned the headlines, {source}", source=self.name
        )

        pass

    def _insert_headlines_to_db(self, headlines):
        self.logger.info(
            "headlines are about to be added to the database, {source}",
            source=self.name,
        )

        pass

    def _get_headlines_from_soup(self, soup):
        pass
