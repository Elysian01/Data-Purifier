"""Performs Automatic Exploratory Data Analysis for NLP datasets."""

import warnings
from datapurifier.decorators import *
from datapurifier.widgets import Widgets
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

import matplotlib.pyplot as plt
# import seaborn as sns
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
import cufflinks as cf
from plotly.offline import iplot
# from sklearn.feature_extraction.text import CountVectorizer

# plt.style.use('ggplot')
py.offline.init_notebook_mode(connected=True)
cf.go_offline()


warnings.filterwarnings("ignore")


# %matplotlib inline


class Nlpeda:
    """Performs Automatic Exploratory Data Analysis for NLP datasets."""

    def __init__(self, df: pd.DataFrame, target: str, explore="basic"):
        self.__set_df_and_target(df, target)
        self.explore = explore
        self.widget = Widgets()

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
            if self.explore == "basic" or self.explore == "advance":
                self.basic_eda()

                print(colored("\nSentiment Analysis:",
                      "red", attrs=["bold"]))
                interact(self.sentiment_analysis, condition=widgets.Checkbox(
                    description="Perform Sentiment Analysis"))

                if self.explore == "advance":
                    self.distribution_plot()

            # print(colored("\nEDA Completed!\n", "green", attrs=["bold"]))
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

    def drop_null_rows(self, x: bool):
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
            self.print_shape()
            self.null_columns_percentage()
            print("Please select to 'drop all null rows', to continue analysis of data.")
            interact(self.drop_null_rows, x=widgets.Checkbox(
                description="Drop all null rows"))

        else:
            self.null_values_present = False
            print(colored(
                "\nCongrats!!, The Dataframe has NO NULL VALUES\n", "green", attrs=["bold"]))

    @conditional_timer
    def __distplot_for_sentiment(self, condition: str) -> None:
        if condition:
            column = "polarity"
            plt.figure(figsize=(6, 4))
            plt.hist(self.df[column], bins=15, color='#0504aa',
                     alpha=0.7, rwidth=0.85)
            plt.xlabel = column
            plt.ylabel = "Count"
            plt.title(column + " Distribution")

    @conditional_timer
    def sentiment_analysis(self, condition: bool):
        if condition:
            self.df["polarity"] = self.df[self.target].apply(
                lambda x: TextBlob(x).sentiment.polarity)
            display(self.df[[self.target, "polarity"]].head())

            interact(self.__distplot_for_sentiment, condition=widgets.Checkbox(
                description="Plot Distribution of Sentiment Analysis"))

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

    def __distplot(self, column: str) -> None:
        if column:
            self.df[column].iplot(kind="hist", xTitle=column,
                                  yTitle="Count", title=column + " Distribution")

    def distribution_plot(self):
        """Plots Distribution Plot of various columns in dataframe"""

        print(colored("\nDistribution Analysis:", "red", attrs=["bold"]))
        col = self.df.columns.tolist()
        col.remove(self.target)
        column_dropdown = self.widget.dropdown(
            options=col, value=None, description="Select Column:")

        items = [column_dropdown]

        hist_plot_ui = widgets.HBox(items)
        output = widgets.interactive_output(self.__distplot, {
                                            'column': column_dropdown})

        display(hist_plot_ui, output)


if __name__ == "__main__":
    df = pd.read_csv("../datasets/twitter16m.csv")
    ae = Nlpeda(df, "tweets")
