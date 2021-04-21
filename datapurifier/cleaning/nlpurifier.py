import re
import spacy
import unicodedata
import pandas as pd
import ipywidgets as widgets
from ipywidgets import GridspecLayout
from ipywidgets import interact, interactive, fixed, interact_manual

from termcolor import colored
from IPython.display import display
from spacy.lang.en.stop_words import STOP_WORDS
from bs4 import BeautifulSoup

from nltk.stem.porter import PorterStemmer

from .contractions import CONTRACTIONS
from datapurifier.decorators import *
from datapurifier.widgets import Widgets
ps = PorterStemmer()


class Nlpurifier:

    def __init__(self: str, df, target: str, spacy_model="en_core_web_sm"):
        self.__set_df_and_target(df, target)
        self.nlp = spacy.load(spacy_model)
        self.widget = Widgets()

        self.purifier_widgets = {}
        self.show_widgets()

    def start_purifying(self, e):

        if self.purifier_widgets["lower"]:
            self.lower()
        if self.purifier_widgets["contraction"]:
            self.contraction_to_expansion()
        if self.purifier_widgets["count_mail"]:
            self.count_emails()
        if self.purifier_widgets["count_urls"]:
            self.count_urls()
        if self.purifier_widgets["word_count"]:
            self.get_word_count()
        if self.purifier_widgets["remove_stop_words"]:
            self.remove_stop_words()
        if self.purifier_widgets["remove_special_and_punct"]:
            self.remove_special_and_punctions()
        if self.purifier_widgets["remove_html"]:
            self.remove_html_tags()
        if self.purifier_widgets["remove_mail"]:
            self.remove_emails()
        if self.purifier_widgets["remove_urls"]:
            self.remove_urls()
        if self.purifier_widgets["remove_spaces"]:
            self.remove_multiple_spaces()
        if self.purifier_widgets["remove_accented"]:
            self.remove_accented_chars()
        # if self.purifier_widgets["remove_words"]:
        #     self.remove_words()
        if self.purifier_widgets["lemma"]:
            self.leammatize()
        if self.purifier_widgets["stem"]:
            self.stemming()

        print(colored("\nCompleted Purifying!\n", "green"))
        print("type <obj>.df to access processed and purified dataframe")

    def show_widgets(self):

        self.lower_widget = self.widget.checkbox(description='Lower all Words')
        self.contraction_widget = self.widget.checkbox(
            description='Contraction to Expansion')
        self.count_mail_widget = self.widget.checkbox(
            description='Count Mails')
        self.count_urls_widget = self.widget.checkbox(description='Count Urls')
        self.word_count_widget = self.widget.checkbox(
            description='Get Word Count')
        self.remove_stop_words_widget = self.widget.checkbox(
            description='Remove Stop Words')
        self.remove_special_and_punct_widget = self.widget.checkbox(
            description='Remove Special Characters and Punctuations')
        self.remove_mail_widget = self.widget.checkbox(
            description='Remove Mails')
        self.remove_html_widget = self.widget.checkbox(
            description='Remove Html Tags')
        self.remove_spaces_widget = self.widget.checkbox(
            description='Remove Multiple Spaces')
        self.remove_accented_widget = self.widget.checkbox(
            description='Remove Accented Characters')
        self.remove_urls_widget = self.widget.checkbox(
            description='Remove Urls')
        self.remove_words_widget = self.widget.checkbox(
            description='Remove Words')
        self.lemma_widget = self.widget.checkbox(description='Leammatize')
        self.stem_widget = self.widget.checkbox(description='Stemming')

        items = [
            [self.lower_widget, self.contraction_widget, self.count_mail_widget],
            [self.count_urls_widget, self.word_count_widget,
                self.remove_stop_words_widget],
            [self.remove_special_and_punct_widget,
                self.remove_mail_widget, self.remove_html_widget],
            [self.remove_urls_widget, self.remove_spaces_widget,
                self.remove_accented_widget],
            [self.remove_words_widget, self.lemma_widget, self.stem_widget]
        ]

        grid_rows = 5
        grid_cols = 3
        grid = GridspecLayout(grid_rows, grid_cols)
        for i in range(len(items)):
            for j in range(len(items[i])):
                grid[i, j] = items[i][j]

        self.grid_output = widgets.interactive_output(
            self.actions, {'lower': self.lower_widget, 'contraction': self.contraction_widget, 'count_mail': self.count_mail_widget,
                           'count_urls': self.count_urls_widget, 'word_count': self.word_count_widget, 'remove_stop_words': self.remove_stop_words_widget,
                           'remove_special_and_punct': self.remove_special_and_punct_widget, 'remove_mail': self.remove_mail_widget,
                           'remove_html': self.remove_html_widget, 'remove_urls': self.remove_urls_widget, 'remove_spaces': self.remove_spaces_widget,
                           'remove_accented': self.remove_accented_widget, 'remove_words': self.remove_words_widget, 'lemma': self.lemma_widget, 'stem': self.stem_widget})

        display(grid)

        start_btn = widgets.Button(description="Start Purifying")
        start_btn.on_click(self.start_purifying)
        display(start_btn)

    def actions(self, lower, contraction, count_mail,
                count_urls, word_count, remove_stop_words,
                remove_special_and_punct, remove_mail, remove_html,
                remove_urls, remove_spaces, remove_accented,
                remove_words, lemma, stem):

        self.purifier_widgets["lower"] = True if lower else False
        self.purifier_widgets["contraction"] = True if contraction else False
        self.purifier_widgets["count_mail"] = True if count_mail else False
        self.purifier_widgets["count_urls"] = True if count_urls else False
        self.purifier_widgets["word_count"] = True if word_count else False
        self.purifier_widgets["remove_stop_words"] = True if remove_stop_words else False
        self.purifier_widgets["remove_special_and_punct"] = True if remove_special_and_punct else False
        self.purifier_widgets["remove_mail"] = True if remove_mail else False
        self.purifier_widgets["remove_html"] = True if remove_html else False
        self.purifier_widgets["remove_urls"] = True if remove_urls else False
        self.purifier_widgets["remove_spaces"] = True if remove_spaces else False
        self.purifier_widgets["remove_accented"] = True if remove_accented else False
        self.purifier_widgets["remove_words"] = True if remove_words else False
        self.purifier_widgets["lemma"] = True if lemma else False
        self.purifier_widgets["stem"] = True if stem else False

    def __set_df_and_target(self, df, target):
        self.df = df
        self.target = target

    def get_text(self):
        self.text = " ".join(self.df[self.target])
        return self.text

    @timer
    def lower(self):
        self.df[self.target] = self.df[self.target].apply(lambda x: x.lower())

    @timer
    def remove_special_and_punctions(self):
        self.df[self.target] = self.df[self.target].apply(
            lambda x: re.sub('[^A-Z a-z 0-9-]+', "", x))

    @timer
    def remove_multiple_spaces(self):
        self.df[self.target] = self.df[self.target].apply(
            lambda x: " ".join(x.split()))

    @timer
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

    @timer
    def contraction_to_expansion(self):
        self.df[self.target] = self.df[self.target].apply(
            lambda x: self.__contraction_to_expansion(x))

    def __remove_accented_chars(self, x: str) -> str:
        x = unicodedata.normalize('NFKD', x).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        return x

    @timer
    def remove_accented_chars(self):
        self.df[self.target] = self.df[self.target].apply(
            lambda x: self.__remove_accented_chars(x))

    @timer
    def remove_stop_words(self):
        self.df[self.target] = self.df[self.target].apply(lambda x: " ".join(
            [word for word in x.split() if word not in STOP_WORDS]))

    @timer
    def count_emails(self):
        self.df['emails'] = self.df[self.target].apply(lambda x: re.findall(
            r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', x))
        self.df["emails_counts"] = self.df["emails"].apply(lambda x: len(x))

    @timer
    def remove_emails(self):
        self.df[self.target] = self.df[self.target].apply(lambda x: re.sub(
            r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', "", x))

    @timer
    def count_urls(self):
        self.df["urls_counts"] = self.df[self.target].apply(lambda x: len(re.findall(
            r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)))

    @timer
    def remove_urls(self):
        self.df[self.target] = self.df[self.target].apply(lambda x: re.sub(
            r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', "", x))

    @timer
    def get_word_count(self):
        text = self.get_text()
        text = text.split()
        self.word_count = pd.Series(text).value_counts()
        print("self.word_count for getting word count series")

    @timer
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

    @timer
    def leammatize(self):
        print(f"""Internally for lemmatization it uses {self.nlp} spacy model,
              to change it please provide `spacy_model='your_model' in constructor`""")
        self.df[self.target] = self.df[self.target].apply(
            lambda x: self.__lemmatize(x))

    def __stemming(self, x):
        return " ".join([ps.stem(word) for word in x.split()])

    @timer
    def stemming(self):
        print("Using Porter Stemmer for stemming")
        self.df[self.target] = self.df[self.target].apply(
            lambda x: self.__stemming(x))


if __name__ == '__main__':
    pass
