"""Performs Automatic Exploratory Data Analysis for NLP datasets."""

import re
import sys
import numpy as np
import pandas as pd
from textblob import TextBlob
from spacy.lang.en.stop_words import STOP_WORDS

from termcolor import colored
from IPython.display import display
from ipywidgets import interact
import ipywidgets as widgets

from datapurifier.widgets import Widgets
from datapurifier.decorators import *

import warnings
warnings.filterwarnings("ignore")


# %matplotlib inline


class Nlpeda:
    """Performs Automatic Exploratory Data Analysis for NLP datasets."""

    def __init__(self, df: pd.DataFrame, target: str, explore="basic"):
        self.__set_df_and_target(df, target)
        self.explore = explore
        self.widget = Widgets()

        self.print_shape()
        self.null_values_present = True
        self.handle_null_values()

        self.__start_analysis()

    def __set_df_and_target(self, df, target):
        self.df = df.copy()
        if target in self.df.columns:
            self.target = target
        else:
            print(colored(
                "Please provide correct `target` column name, containing only textual data for analysis", "red", attrs=["bold"]))
            sys.exit(1)

    def __start_analysis(self):
        if not self.null_values_present:
            if self.explore == "basic":
                self.basic_eda()

            print(colored("\nEDA Completed!\n", "green", attrs=["bold"]))
            print("type <obj>.df to access explored dataframe")

    def print_shape(self):
        print(
            colored(f"Dataframe contains {self.df.shape[0]} rows and {self.df.shape[1]} columns\n", "blue", attrs=["bold"]))

    def get_avg_word_len(self, x: str) -> float:
        words = x.split()
        word_length = 0
        for word in words:
            word_length = word_length + len(word)
        return word_length/len(words)

    @exception_handler
    def count_emails(self):
        self.df['emails'] = self.df[self.target].apply(lambda x: re.findall(
            r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', x))
        self.df["emails_counts"] = self.df["emails"].apply(lambda x: len(x))

    @exception_handler
    def count_urls(self):
        self.df["urls_counts"] = self.df[self.target].apply(lambda x: len(re.findall(
            r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)))

    @exception_handler
    def null_columns_percentage(self) -> pd.DataFrame:
        '''
        Prints Null Information of dataframe,i.e. only the number of rows having null values and their null percentage
        '''
        print("\nNull Information of Dataframe: ")
        self.null_df = pd.DataFrame(self.df.isnull().sum()).reset_index()
        self.null_df.columns = ["column_name", "null_rows"]
        self.null_df["null_percentage"] = self.null_df["null_rows"] * \
            100 / self.df.shape[0]
        self.null_df = self.null_df[self.null_df["null_percentage"] != 0].sort_values(
            "null_percentage", ascending=False).reset_index(drop=True)

        display(self.null_df)
        return self.null_df

    def drop_null_rows(self, x):
        """Drops rows having [' ', 'NULL', np.nan] values """
        if x:
            total_null_rows = self.df[self.target].isin(
                [' ', 'NULL', np.nan]).sum()
            if total_null_rows > 0:
                print("Dropping rows having [' ', 'NULL', numpy.nan] values")
                self.df.dropna(inplace=True)
                self.df.reset_index(drop=True, inplace=True)
                self.null_values_present = False
                print(
                    colored(f"Total Null rows dropped: {total_null_rows}\n", "red", attrs=["bold"]))
                self.__start_analysis()
            else:
                print(colored("There is no null rows present.\n", "green"))

    @exception_handler
    def handle_null_values(self) -> bool:
        # Null value analysis
        if self.df.isnull().sum().sum() > 0:
            self.null_columns_percentage()
            print("Please select to 'drop all null rows', to continue analysis of data.")
            interact(self.drop_null_rows, x=widgets.Checkbox(
                description="Drop all null rows"))

        else:
            self.null_values_present = False
            print(colored(
                "\nCongrats!!, The Dataframe has NO NULL VALUES\n", "green", attrs=["bold"]))

    @timer_and_exception_handler
    def basic_eda(self):
        self.print_shape()

        # Word Count
        self.df["word_counts"] = self.df[self.target].apply(
            lambda x: len(str(x).split()))
        # Char Count
        self.df["char_counts"] = self.df[self.target].apply(
            lambda x: len(str(x)))
        # Average Word Length
        self.df["average_word_lengths"] = self.df[self.target].apply(
            lambda x: self.get_avg_word_len(x))
        # Stop Words Count
        self.df["stop_words_counts"] = self.df[self.target].apply(lambda x: len(
            [word for word in x.split() if word in STOP_WORDS]))
        # Uppercase Word Count
        self.df["upper_case_word_counts"] = self.df[self.target].apply(lambda x: len(
            [word for word in x if word.isupper() and len(x) > 3]))

        display(self.df.head())


if __name__ == "__main__":
    df = pd.read_csv("../datasets/twitter16m.csv")
    ae = Nlpeda(df, "tweets")
