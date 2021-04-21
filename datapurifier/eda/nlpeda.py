"""Performs Automatic Exploratory Data Analysis for NLP datasets."""

from datapurifier.widgets import Widgets
from ipywidgets.widgets import widget
import numpy as np
import pandas as pd
import re
# from textblob import TextBlob
from spacy.lang.en.stop_words import STOP_WORDS

from termcolor import colored
from IPython.display import display
import ipywidgets as widgets
import warnings
warnings.filterwarnings("ignore")


# %matplotlib inline


class Nlpeda:
    """Performs Automatic Exploratory Data Analysis for NLP datasets."""

    def __init__(self, df, target: str, explore="basic"):
        self.__set_df_and_target(df, target)
        if explore == "basic":
            self.basic_eda(df, target)

    def __set_df_and_target(self, df, target):
        self.df = df
        self.target = target

    def get_avg_word_len(self, x: str) -> float:
        words = x.split()
        word_length = 0
        for word in words:
            word_length = word_length + len(word)
        return word_length/len(words)

    def count_emails(self, df, target):
        df['emails'] = df[self.target].apply(lambda x: re.findall(
            r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', x))
        df["emails_counts"] = df["emails"].apply(lambda x: len(x))

    def count_urls(self, df, target):
        df["urls_counts"] = df[self.target].apply(lambda x: len(re.findall(
            r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)))

    def basic_eda(self, df, target):
        self.__set_df_and_target(df, target)
        # Word Count
        df["word_counts"] = df[self.target].apply(
            lambda x: len(str(x).split()))
        # Char Count
        df["char_counts"] = df[self.target].apply(lambda x: len(str(x)))
        # Average Word Length
        df["average_word_lengths"] = df[self.target].apply(
            lambda x: self.get_avg_word_len(x))
        # Stop Words Count
        df["stop_words_counts"] = df[self.target].apply(lambda x: len(
            [word for word in x.split() if word in STOP_WORDS]))
        # Uppercase Word Count
        df["upper_case_word_counts"] = df[self.target].apply(lambda x: len(
            [word for word in x if word.isupper() and len(x) > 3]))

        df.head()


if __name__ == "__main__":
    df = pd.read_csv("../datasets/twitter16m.csv")
    ae = Nlpeda(df, "tweets")
