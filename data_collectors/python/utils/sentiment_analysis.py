import nltk

nltk.download("vader_lexicon")

from nltk.sentiment.vader import SentimentIntensityAnalyzer


class SentimentAnalysis:
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()
        self.name = "vader"

    def analyze_sentense(self, sentence):
        return self.sid.polarity_scores(sentence)

    def is_sentence_positive(self, sentence):
        pass

    def print_score_for_sentence(self, sentence):
        ss = self.analyze_sentense(sentence)
        for k in sorted(ss):
            print("{0}: {1}, ".format(k, ss[k]), end="")
        print()

    def print_score(self, ss):
        for k in sorted(ss):
            print("{0}: {1}, ".format(k, ss[k]), end="")
        print()

if __name__ == '__main__':
    yeet = SentimentAnalysis()
    yeet.print_score_for_sentence("the market is bullish")