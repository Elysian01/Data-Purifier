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
from wordcloud import WordCloud
from ipywidgets import interact
import ipywidgets as widgets

import matplotlib.pyplot as plt
# import seaborn as sns
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
import cufflinks as cf
from plotly.offline import iplot
from sklearn.feature_extraction.text import CountVectorizer

from datapurifier.utils import *

# plt.style.use('ggplot')
py.offline.init_notebook_mode(connected=True)
cf.go_offline()


warnings.filterwarnings("ignore")


# %matplotlib inline


class Nlpeda:
    """Performs Automatic Exploratory Data Analysis for NLP datasets."""

    def __init__(self, df: pd.DataFrame, target: str, analyse="basic"):
        self._set_df_and_target(df, target)
        self.analyse = analyse
        self.widget = Widgets()

        self.word_list = []
        self.word_count = None
        self.null_values_present = True
        self.handle_null_values()

        self._start_analysis()

    def _set_df_and_target(self, df, target):
        self.df = df.copy()
        if target in self.df.columns:
            self.target = target
        else:
            print_in_red(
                "Please provide correct `target` column name, containing only textual data for analysis")
            sys.exit(1)

    def _start_analysis(self):
        if not self.null_values_present:

            if self.analyse == "basic":
                self.basic_eda()

                print(colored("\nSentiment Analysis:",
                      "red", attrs=["bold"]))
                interact(self.sentiment_analysis, condition=widgets.Checkbox(
                    description="Perform Sentiment Analysis"))

                self.distribution_plot()

                print(colored("\nEDA Completed!\n", "green", attrs=["bold"]))
                print("type <obj>.df to access explored dataframe")

            if self.analyse == "word":
                self.preprocess_text()
                self.find_word_count()
                self.plot_wordcloud()

                self.unigram_df = pd.DataFrame()
                self.bigram_df = pd.DataFrame()
                self.trigram_df = pd.DataFrame()
                self.unigram_statistics()
                self.bigram_statistics()
                self.trigram_statistics()
                self.ngram_plot()

    def preprocess_text(self):
        """Generates a complete documnet containing only text, list of all words, and count of each word """
        self.text = " ".join(self.df[self.target])
        self.word_list = self.text.split()
        self.word_count = pd.Series(self.word_list).value_counts()
        return self.text, self.word_list, self.word_count

    def print_shape(self):
        print_in_blue(
            f"Dataframe contains {self.df.shape[0]} rows and {self.df.shape[1]} columns\n")

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
                print_in_red(f"Total Null rows dropped: {total_null_rows}\n")
                self._start_analysis()
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
    def _distplot_for_sentiment(self, condition: str) -> None:
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

            interact(self._distplot_for_sentiment, condition=widgets.Checkbox(
                description="Plot Sentiment Distribution"))

    @timer
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

        display(self.df.head())

    def _distplot(self, column: str) -> None:
        if column:
            self.df[column].iplot(kind="hist", xTitle=column,
                                  yTitle="Count", title=column + " Distribution")

    def distribution_plot(self):
        """Plots Distribution Plot of various columns in dataframe"""

        print_in_red("\nDistribution Analysis:")
        col = self.df.columns.tolist()
        col.remove(self.target)
        column_dropdown = self.widget.dropdown(
            options=col, value=None, description="Column: ")

        items = [column_dropdown]

        hist_plot_ui = widgets.HBox(items)
        output = widgets.interactive_output(self._distplot, {
                                            'column': column_dropdown})

        display(hist_plot_ui, output)

    """
    -------------------------------------------------------------------
    Word Analysis Start
    -------------------------------------------------------------------
    """

    def _perform_word_count(self, word):
        if word:
            try:
                print(
                    f"The word '{word}' has occured {self.word_count[word]} times in dataset.")
            except:
                print("No such word in dataset.")

    def find_word_count(self):
        print(colored("Enter Word and find its count: ",
              "blue", attrs=["bold"]))
        interact(self._perform_word_count, word=widgets.Text(
            placeholder="Enter Your word here", description="Word"))

    @exception_handler
    def _perform_wordcloud_visualization(self, condition):
        if condition:
            print("Please wait plotting wordcloud...")
            wc = WordCloud(width=1000, height=400).generate(str(self.text))
            plt.axis("off")
            plt.imshow(wc)

    @exception_handler
    def plot_wordcloud(self):
        print_in_blue("Plot Wordcloud: ")
        interact(self._perform_wordcloud_visualization, condition=widgets.Checkbox(
            description="Plot Wordcloud"))

    def get_top_n_words(self, x, n: int, ngram_range: tuple, include_stop_words: bool = False):
        if include_stop_words:
            vec = CountVectorizer(ngram_range=ngram_range).fit(x)
        else:
            vec = CountVectorizer(ngram_range=ngram_range,
                                  stop_words="english").fit(x)

        bow = vec.transform(x)
        sum_words = bow.sum(axis=0)
        word_frequence = [(word, sum_words[0, idx])
                          for word, idx in vec.vocabulary_.items()]
        word_frequence = sorted(
            word_frequence, key=lambda x: x[1], reverse=True)
        return word_frequence[:n]

    # Unigram Analysis

    def _unigram_interaction(self, top_unigram: int) -> None:
        self.top_unigram = top_unigram

    def _unigram_stop_word_interaction(self, condition: bool):
        self.unigram_stop_words = condition

    def _start_unigram(self, e):
        print(
            f"Please wait starting unigram word analysis for getting top {self.top_unigram} words...")

        self.unigram = self.get_top_n_words(self.df[self.target], self.top_unigram,
                                            (1, 1), self.unigram_stop_words)
        self.unigram_df = pd.DataFrame(self.unigram, columns=[
            "unigram", "frequence"])

        # display(self.unigram_df.head(3))
        print("Analysis Completed!, to view whole dataframe type 'obj.unigram_df'")

    def _perform_unigram(self, condition: bool):
        if condition:
            top_unigram_widget = self.widget.int_slider(
                minimum=1, maximum=100, step=1, value=10, description="Word Range:")

            interact(self._unigram_stop_word_interaction, condition=widgets.Checkbox(
                description="Include Stop words in analysis"))

            widgets.interactive_output(
                self._unigram_interaction, {'top_unigram': top_unigram_widget})

            display(top_unigram_widget)

            unigram_button = widgets.Button(
                description='Start Analysis',
                tooltip='Start Unigram Analysis',
                button_style='info'
            )

            unigram_button.on_click(self._start_unigram)
            display(unigram_button)

    @exception_handler
    def unigram_statistics(self):
        print_in_blue("Unigram Analysis: ")
        interact(self._perform_unigram, condition=widgets.Checkbox(
            description="Perform Unigram"))

    # Bigram Analysis

    def _bigram_interaction(self, top_bigram: int) -> None:
        self.top_bigram = top_bigram

    def _bigram_stop_word_interaction(self, condition: bool):
        self.bigram_stop_words = condition

    def _start_bigram(self, e):
        print(
            f"Please wait starting Bigram word analysis for getting top {self.top_bigram} words...")

        self.bigram = self.get_top_n_words(self.df[self.target], self.top_bigram,
                                           (2, 2), self.bigram_stop_words)
        self.bigram_df = pd.DataFrame(self.bigram, columns=[
            "bigram", "frequence"])

        # display(self.bigram_df.head(3))
        print("Analysis Completed!, to view whole dataframe type 'obj.bigram_df'")

    def _perform_bigram(self, condition: bool):

        if condition:
            top_bigram_widget = self.widget.int_slider(
                minimum=1, maximum=100, step=1, value=10, description="Word Range:")

            interact(self._bigram_stop_word_interaction, condition=widgets.Checkbox(
                description="Include Stop words in analysis"))

            widgets.interactive_output(
                self._bigram_interaction, {'top_bigram': top_bigram_widget})

            display(top_bigram_widget)

            bigram_button = widgets.Button(
                description='Start Analysis',
                tooltip='Start Bigram Analysis',
                button_style='info'
            )

            bigram_button.on_click(self._start_bigram)
            display(bigram_button)

    @exception_handler
    def bigram_statistics(self):
        print_in_blue("Bigram Analysis: ")
        interact(self._perform_bigram, condition=widgets.Checkbox(
            description="Perform Bigram"))

    # Trigram Analysis

    def _trigram_interaction(self, top_trigram: int) -> None:
        self.top_trigram = top_trigram

    def _trigram_stop_word_interaction(self, condition: bool):
        self.trigram_stop_words = condition

    def _start_trigram(self, e):
        print(
            f"Please wait starting Trigram word analysis for getting top {self.top_trigram} words...")

        self.trigram = self.get_top_n_words(self.df[self.target], self.top_trigram,
                                            (3, 3), self.trigram_stop_words)
        self.trigram_df = pd.DataFrame(self.trigram, columns=[
            "trigram", "frequence"])

        # display(self.trigram_df.head(3))
        print("Analysis Completed!, to view whole dataframe type 'obj.trigram_df'")

    def _perform_trigram(self, condition: bool):
        if condition:
            top_trigram_widget = self.widget.int_slider(
                minimum=1, maximum=100, step=1, value=10, description="Word Range:")

            interact(self._trigram_stop_word_interaction, condition=widgets.Checkbox(
                description="Include Stop words in analysis"))

            widgets.interactive_output(
                self._trigram_interaction, {'top_trigram': top_trigram_widget})

            display(top_trigram_widget)

            trigram_button = widgets.Button(
                description='Start Analysis',
                tooltip='Start Trigram Analysis',
                button_style='info'
            )

            trigram_button.on_click(self._start_trigram)
            display(trigram_button)

    @exception_handler
    def trigram_statistics(self):
        print_in_blue("Trigram Analysis: ")
        interact(self._perform_trigram, condition=widgets.Checkbox(
            description="Perform Trigram"))

    @exception_handler
    def _ngram_plot(self, df):
        if df == "Unigram":
            plot_df = self.unigram_df.set_index("unigram")
            plot_df.iplot(kind="bar", xTitle="Unigram",
                          yTitle="Frequence", title="Top Unigram words", dimensions=(1000, 350))
        if df == "Bigram":
            plot_df = self.bigram_df.set_index("bigram")
            plot_df.iplot(kind="bar", xTitle="Bigram",
                          yTitle="Frequence", title="Top Bigram words", dimensions=(1000, 350))
        if df == "Trigram":
            plot_df = self.trigram_df.set_index("trigram")
            plot_df.iplot(kind="bar", xTitle="Trigram",
                          yTitle="Frequence", title="Top Trigram words", dimensions=(1000, 350))

    @exception_handler
    def _perform_ngram(self, condition: bool):
        print("To refresh selection, please uncheck and check `Start Plotting` checkbox")
        if condition:
            df = []
            if not self.unigram_df.empty:
                df.append("Unigram")
            if not self.bigram_df.empty:
                df.append("Bigram")
            if not self.trigram_df.empty:
                df.append("Trigram")

            df_dropdown = self.widget.dropdown(
                options=df, value=None, description="N-Grams: ")

            items = [df_dropdown]

            hist_plot_ui = widgets.HBox(items)
            output = widgets.interactive_output(self._ngram_plot, {
                                                'df': df_dropdown})

            display(hist_plot_ui, output)

    @exception_handler
    def ngram_plot(self):
        print_in_blue("Plot Ngram Plots: ")
        interact(self._perform_ngram, condition=widgets.Checkbox(
            description="Start Plotting"))


if __name__ == "__main__":
    df = pd.read_csv("../datasets/twitter16m.csv")
    ae = Nlpeda(df, "tweets")
