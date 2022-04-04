from utils.sentiment_analysis import SentimentAnalysis
from utils.translator import Translator
from utils.scraper import call_url_get_bs4
from utils.companines import CompaniesInHeadline
from bs4 import BeautifulSoup
import logging
import pandas as pd
import time
import re
from datetime import datetime, timedelta
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


class Boersen:
    def __init__(
        self,
        analyzer: SentimentAnalysis,
        translator: Translator,
        company_headline: CompaniesInHeadline,
        logger,
    ):
        self.analyzer = analyzer
        self.translator = translator
        self.company_headline = company_headline
        self.logger = logger
        self.name = "boersen"
        self.base_url = "https://api.borsen.dk/nyheder/side"

    def start(self):
        articels = self._get_article_dataframe()
        i = 1
        while True:
            try:
                self.logger.info(
                    "about to fetch for page {page} from {source}",
                    page=i,
                    source=self.name,
                )
                print(f"page {i}")
                soup = call_url_get_bs4(
                    f"{self.base_url}/{i}",
                    cookies=self._get_cookies(),
                    headers=self._get_headers(),
                )
                page = soup.find_all("div", {"class": "col offset-md-2 body"})
                if len(page) == 0 or i == 10:
                    break
                else:
                    articels = self._parse_articels_in_page(page)
                    i += 1
                    break
            except Exception as e:
                print(f"exception at {i} - {str(e)}")
                time.sleep(5)
        articels.to_csv("this.csv")

    def _parse_articels_in_page(self, page):
        articels = self._get_article_dataframe()
        for link in page:
            while True:
                try:
                    if not link.a == None:
                        url = link.a.get("href")
                        inner_soup = call_url_get_bs4(
                            url,
                            cookies=self._get_cookies(),
                            headers=self._get_headers(),
                        )

                        publish_date = inner_soup.find_all(
                            "span", {"class": "published"}
                        )

                        if self._date_with_correct_format_exists(publish_date):
                            articels = self._parse_data_to_dataframe(
                                articels, link, publish_date, url
                            )

                        else:
                            print(f"error - {url} - {publish_date}")
                        break
                except Exception as e:
                    print(f"Error {str(e)}")
                    print("retrying in a few seconds")
                    time.sleep(5)
        return articels

    def _parse_data_to_dataframe(self, articels, link, publish_date, url):
        headline = link.a.get_text()
        date = self._get_date(publish_date)

        companies = self.company_headline.get_companies_in_headlinen(headline)

        target_headline = self.translator.translate_sentence(headline)

        score = self.analyzer.analyze_sentense(target_headline)

        return self._add_row(
            articels, date, headline, url, companies, target_headline, score
        )

    def _get_headers(self):
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36",
        }

    def _get_cookies(self):
        return {}

    def _get_date(self, mydivs):
        date = mydivs[0].get_text()
        date = (
            date.replace(".", " ")
            .replace(":", " ")
            .replace("KL", " ")
            .replace("    ", " ")
            .replace("  ", " ")
        )
        return self._parse_date(date)

    def _date_with_correct_format_exists(self, mydivs):
        return len(mydivs) > 0 and re.findall(
            "[0-9]*\. [a-z]+ [0-9]* KL. [0-9]*:[0-9]*", mydivs[0].get_text()
        )

    def _add_row(
        self, articels, date, headline, url, companies, target_headline, score
    ):
        row = pd.DataFrame(
            {
                "release_date": [date],
                "source_headline": [headline],
                "source_language": [self.translator.source_langeuage],
                "target_language": [self.translator.target_language],
                "target_headline": [target_headline],
                "url": [url],
                "companies": [",".join([str(x) for x in companies])],
                "neg": [score["neg"]],
                "pos": [score["pos"]],
                "compound": [score["compound"]],
                "neu": [score["neu"]],
            }
        )
        return pd.concat([articels, row], ignore_index=True, axis=0)

    def _get_article_dataframe(self):
        return pd.DataFrame(
            columns=[
                "release_date",
                "source_headline",
                "target_headline",
                "source_language",
                "target_language",
                "neg",
                "pos",
                "neu",
                "compound",
                "url",
                "companies",
            ]
        )

    def _parse_date(self, date):
        split_date = date.split(" ")
        day = int(split_date[0])
        month = int(self._get_months()[split_date[1]])
        year = int(split_date[2])
        hour = int(split_date[3])
        minute = int(split_date[4])

        return datetime(year=year, month=month, day=day, hour=hour, minute=minute)

    def _get_months(self):
        return {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "maj": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "okt": 10,
            "nov": 11,
            "dec": 12,
        }
