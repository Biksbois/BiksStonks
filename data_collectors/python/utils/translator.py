from deep_translator import GoogleTranslator


class Translator:
    def __init__(self, source_language="da", target_language="en"):
        self.gt = GoogleTranslator(source=source_language, target=target_language)
        self.source_langeuage = source_language
        self.target_language = target_language
        self.name = "google translate"

    def translate_sentence(self, sentence):
        return self.gt.translate(sentence)

    def supported_languages(self):
        return self.gt.get_supported_languages()
