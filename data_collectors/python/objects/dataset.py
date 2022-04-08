class SentimentDataset:
    def __init__(
        self, translator, source_language, target_language, source, url, category
    ):
        self.translator = translator
        self.source_language = source_language
        self.target_language = target_language
        self.source = source
        self.url = url
        self.category = category

    def __str__(self):
        return (
            self.translator
            + self.source_language
            + self.target_language
            + self.source
            + self.url
            + self.category
        )

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return (
            self.translator == other.translator
            and self.source_language == other.source_language
            and self.target_language == other.target_language
            and self.source == other.source
            and self.url == other.url
            and self.category == other.category
        )
