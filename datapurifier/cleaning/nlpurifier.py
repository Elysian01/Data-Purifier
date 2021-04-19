import pandas as pd
import re
import ipywidgets as widgets
from termcolor import colored
from IPython.display import display
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from bs4 import BeautifulSoup
import unicodedata

from nltk.stem.porter import PorterStemmer

from .contractions import CONTRACTIONS
from datapurifier.widgets import Widgets
ps = PorterStemmer()


class Nlpurifier:

    def __init__(self: str, df, target: str, spacy_model="en_core_web_sm"):
        self.__set_df_and_target(df, target)
        self.nlp = spacy.load(spacy_model)
        self.widget = Widgets()

        self.__set_functions()
        self.show_widgets()

    def show_widgets(self):

        lower = self.widget.checkbox(description='Lower all Words')
        contraction = self.widget.checkbox(
            description='Contraction to Expansion')
        count_mail = self.widget.checkbox(description='Count Mails')
        count_urls = self.widget.checkbox(description='Count Urls')
        word_count = self.widget.checkbox(description='Get Word Count')
        remove_stop_words = self.widget.checkbox(
            description='Remove Stop Words')
        remove_special_and_punct = self.widget.checkbox(
            description='Remove Special Characters and Punctuations')
        remove_mail = self.widget.checkbox(description='Remove Mails')
        remove_html = self.widget.checkbox(
            description='Remove Html Tags')
        remove_spaces = self.widget.checkbox(
            description='Remove Multiple Spaces')
        remove_accented = self.widget.checkbox(
            description='Remove Accented Characters')
        remove_urls = self.widget.checkbox(description='Remove Urls')
        remove_words = self.widget.checkbox(description='Remove Words')
        lemma = self.widget.checkbox(description='Leammatize')
        stem = self.widget.checkbox(description='Stemming')

        items = [lower, contraction, count_mail,
                 count_urls, word_count, remove_stop_words,
                 remove_special_and_punct, remove_mail, remove_html,
                 remove_spaces, remove_accented, remove_urls,
                 remove_words, lemma, stem
                 ]
        nlp_cleaner_ui = widgets.HBox(items)

        out = widgets.interactive_output(
            self.actions, {'lower': lower})

        display(nlp_cleaner_ui, out)

    def actions(self, lower):
        pass

    def __set_df_and_target(self, df, target):
        self.df = df
        self.target = target

    def get_text(self):
        self.text = " ".join(self.df[self.target])
        return self.text

    def lower(self):
        self.df[self.target] = self.df[self.target].apply(lambda x: x.lower())

    def remove_special_and_punctions(self):
        self.df[self.target] = self.df[self.target].apply(
            lambda x: re.sub('[^A-Z a-z 0-9-]+', "", x))

    def remove_multiple_spaces(self):
        self.df[self.target] = self.df[self.target].apply(
            lambda x: " ".join(x.split()))

    def remove_html_tags(self):
        self.df[self.target] = self.df[self.target].apply(
            lambda x: BeautifulSoup(x, 'html.parser').get_text())

    def __contraction_to_expansion(self, x: str) -> str:
        if type(x) is str:
            for key in CONTRACTIONS:
                value = CONTRACTIONS[key]
                x = x.replace(key, value)
            return x
        else:
            return x

    def contraction_to_expansion(self):
        self.df[self.target] = self.df[self.target].apply(
            lambda x: self.__contraction_to_expansion(x))

    def __remove_accented_chars(self, x: str) -> str:
        x = unicodedata.normalize('NFKD', x).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        return x

    def remove_accented_chars(self):
        self.df[self.target] = self.df[self.target].apply(
            lambda x: self.__remove_accented_chars(x))

    def remove_stop_words(self):
        self.df[self.target] = self.df[self.target].apply(lambda x: " ".join(
            [word for word in x.split() if word not in STOP_WORDS]))

    def count_emails(self):
        self.df['emails'] = self.df[self.target].apply(lambda x: re.findall(
            r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', x))
        self.df["emails_counts"] = self.df["emails"].apply(lambda x: len(x))

    def remove_emails(self):
        self.df[self.target] = self.df[self.target].apply(lambda x: re.sub(
            r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', "", x))

    def count_urls(self):
        self.df["urls_counts"] = self.df[self.target].apply(lambda x: len(re.findall(
            r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)))

    def remove_urls(self):
        self.df[self.target] = self.df[self.target].apply(lambda x: re.sub(
            r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', "", x))

    def get_word_count(self):
        text = self.get_text()
        text = text.split()
        self.word_count = pd.Series(text).value_counts()
        print("self.word_count for getting word count series")

    def remove_words(self, word_list):
        """Removes words which are in word list

        Args:
            word_list (list): List of words to be removed.
        """
        self.df[self.target] = self.df[self.target].apply(lambda x: " ".join(
            [word for word in x.split() if word not in word_list]))

    def __lemmatize(self, x: str) -> str:
        """Uses spacy library to lemmatize words.

        Args:
            x (str): sentence
        """
        doc = self.nlp(x)
        lem = ""
        for token in doc:
            lem += token.lemma_ + " "
        return lem

    def leammatize(self):
        print(f"""Internally for lemmatization it uses {self.nlp} spacy model,
              to change it please provide `spacy_model='your_model' in constructor`""")
        self.df[self.target] = self.df[self.target].apply(
            lambda x: self.__lemmatize(x))

    def __stemming(self, x):
        return " ".join([ps.stem(word) for word in x.split()])

    def stemming(self):
        print("Using Porter Stemmer for stemming")
        self.df[self.target] = self.df[self.target].apply(
            lambda x: self.__stemming(x))

    def __set_functions(self):
        self.nl_cleaner_functions = {
            'Lower all Words': self.lower,
            'Contraction to Expansion': self.contraction_to_expansion,
            'Count Mails': self.count_emails,
            'Count Urls': self.count_urls,
            'Get Word Count': self.get_word_count,
            'Remove Stop Words': self.remove_stop_words,
            'Remove Special and Punctions': self.remove_special_and_punctions,
            'Remove Mails': self.remove_emails,
            'Remove Html Tags': self.remove_html_tags,
            'Remove Multiple Spaces': self.remove_multiple_spaces,
            'Remove Accented Characters': self.remove_accented_chars,
            'Remove Urls': self.remove_urls,
            'Remove Words': self.remove_words,
            'Leammatize': self.leammatize,
            'Stemming': self.stemming
        }


if __name__ == '__main__':
    pass
