from fnmatch import translate
from utils.sentiment_analysis import SentimentAnalysis
from utils.translator import Translator
from utils.scraper import call_url_get_bs4
from utils.companines import CompaniesInHeadline
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime, timedelta
from tqdm import tqdm
from tqdm import trange
import requests
import logging
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

logging.basicConfig(filename="log.log", level=logging.DEBUG)


class Boersen:
    def __init__(
        self,
        analyzer: SentimentAnalysis,
        translator: Translator,
        company_headline: CompaniesInHeadline,
        db,
    ):
        self.analyzer = analyzer
        self.translator = translator
        self.company_headline = company_headline
        self.db = db
        self.name = "boersen"
        self.base_url = "https://api.borsen.dk/nyheder/side"

    def start(self):
        i = 1433
        while True:
            articels = self._get_article_dataframe()
            try:
                print(f"\n---\nabout to fetch for page {i} from {self.name}")
                soup = call_url_get_bs4(
                    f"{self.base_url}/{i}",
                    cookies=self._get_cookies(),
                    headers=self._get_headers(),
                )

                page = soup.find_all("div", {"class": "col offset-md-2 body"})
                categories = soup.find_all("div", {"class": "d-none d-md-block"})

                if len(page) == 0:
                    break
                else:
                    articels = self._parse_articels_in_page(page, categories)
                    self.db.dataframe_to_db(
                        articels,
                        "upsert_sentiment",
                        "upsert_sentiment_dataset",
                        self.translator.name,
                        self.base_url,
                        self.name,
                        self._get_description(),
                    )
                    i += 1
            except Exception as e:
                logging.info(str(e))
                # print("----")
                # print(str(e))
                time.sleep(10)

    def _get_description(self):
        return (
            f"{self.name} headlines, translated from {self.translator.source_langeuage} to {self.translator.target_language} using {self.translator.name}, and analyzed using {self.analyzer.name}",
        )

    def _parse_articels_in_page(self, page, categories):
        articels = self._get_article_dataframe()
        error_urls = []
        max_tries = 3
        t = trange(len(page), desc="running", leave=True)
        for i in t:
            link = page[i]
            error_count = 0
            while True:
                if error_count > max_tries:
                    break

                try:
                    t.set_description("running")
                    t.refresh()

                    if not link.a == None:
                        url = link.a.get("href")
                        inner_soup = call_url_get_bs4(
                            url,
                            cookies=self._get_cookies(),
                            headers=self._get_headers(),
                        )

                        publish_date1 = inner_soup.find_all(
                            "span", {"class": "published"}
                        )

                        publish_date2 = inner_soup.find_all(
                            "div",
                            {
                                "class": "label label-M font__gtamerica_condensed--medium"
                            },
                        )

                        if self._date_with_correct_format_exists(publish_date1):
                            articels = self._parse_data_to_dataframe(
                                articels, link, publish_date1, url, categories[i]
                            )
                        elif self._date_with_correct_format_exists(publish_date2):
                            articels = self._parse_data_to_dataframe(
                                articels, link, publish_date2, url, categories[i]
                            )
                        else:
                            error_urls.append(url)
                    break
                except Exception as e:
                    error_count += 1
                    desc = "sleeping..." if error_count == 0 else f"Trying last time ({error_count})" if error_count == max_tries else f"try no {error_count}"
                    t.set_description(desc)
                    t.refresh()
                    logging.info(str(e))
                    # print("---")
                    # print(str(e))
                    time.sleep(10)

        if len(error_urls) == 0:
            print("\nNo URL caused an error on this page\n")
        else:
            print(f"\n{len(error_urls)} caused an error on this page\n")
            self.append_list_to_text_file(error_urls, "error_urls.txt")

        return articels

    def append_list_to_text_file(self, list, file_name):
        with open(file_name, "a") as file:
            for item in list:
                file.write("%s\n" % item)

    def _parse_data_to_dataframe(self, articels, link, publish_date, url, category):
        headline = link.a.get_text()
        date = self._get_date(publish_date)

        companies = self.company_headline.get_companies_in_headlinen(headline)

        target_headline = self.translator.translate_sentence(headline)

        category = category.get_text()

        score = self.analyzer.analyze_sentense(target_headline)

        return self._add_row(
            articels, date, headline, url, companies, target_headline, score, category
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
        return self._parse_date(date.lstrip())

    def _date_with_correct_format_exists(self, mydivs):
        return len(mydivs) > 0 and re.findall(
            "[0-9]*\. [a-z]+[\.]* [0-9]* KL.[ ]*[0-9]*:[0-9]*", mydivs[0].get_text()
        )

    def _add_row(
        self, articels, date, headline, url, companies, target_headline, score, category
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
                "category": [category],
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
                "category",
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
