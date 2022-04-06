from unittest.mock import sentinel
from utils.sentiment_analysis import SentimentAnalysis
from utils.companines import CompaniesInHeadline
from utils.translator import Translator
from sentiment_sources.boersen import Boersen
from utils.logger import initialize_logger
import logging
import time

if __name__ == "__main__":
    name = "datacollector"

    initialize_logger()
    logger = logging.getLogger(name)

    logger.info("Starting...")

    analyzer = SentimentAnalysis()
    danish_translator = Translator(source_language="da", target_language="en")
    company_in_headline = CompaniesInHeadline()

    boersen = Boersen(analyzer, danish_translator, company_in_headline, logger)

    news_source = [boersen]

    for source in news_source:
        try:
            source.start()
        except Exception as e:
            logger.exception(
                "source {source} msg: {msg}", source=source.name, msg=str(e)
            )

    danish_headline = (
        "Ukraine-krig slynger danske virksomheder ud i kaos på globale markeder"
    )

    english_headline = danish_translator.translate_sentence(danish_headline)

    print(english_headline)
    score = analyzer.analyze_sentense(english_headline)
    analyzer.print_score(score)

    logger.info("ending...")
    time.sleep(2)
