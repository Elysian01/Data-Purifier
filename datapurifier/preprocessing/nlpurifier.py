import re
import sys
import spacy
import unidecode
import unicodedata
import numpy as np
import pandas as pd
import ipywidgets as widgets
from ipywidgets import interact
from ipywidgets import GridspecLayout

from termcolor import colored
from IPython.display import display
from spacy.lang.en.stop_words import STOP_WORDS
from bs4 import BeautifulSoup

from nltk.stem.porter import PorterStemmer

from datapurifier.preprocessing.data_dict.contractions import CONTRACTIONS
from datapurifier.preprocessing.data_dict.emoticons import EMOTICONS
from datapurifier.preprocessing.data_dict.emoji import UNICODE_EMO
from datapurifier.decorators import *
from datapurifier.widgets import Widgets
from datapurifier.utils import *
ps = PorterStemmer()


def _auto_detect_noise_in_textual_data():
    """Auto-Detection, of emojis, links, emails, and accented characters"""
    pass


def NLAutoPurifier(df: pd.DataFrame, target: str, representation=None):
    """
    1. Automatically Pre-process text-data by pipeline.
    2. Auto-Detection, of emojis, links, emails, and accented characters
    3. Represent words into their vector format

    Default pipeline for step-1:
    1. :meth:`datapurifier.Nlpurifier.drop_null_rows`
    2. :meth:`datapurifier.Nlpurifier.lower`
    3. :meth:`datapurifier.Nlpurifier.remove_numbers`
    4. :meth:`datapurifier.Nlpurifier.remove_html_tags`
    5. :meth:`datapurifier.Nlpurifier.convert_accented_chars_to_normal`
    6. :meth:`datapurifier.Nlpurifier.remove_punctuations`
    7. :meth:`datapurifier.Nlpurifier.remove_stop_words`
    8. :meth:`datapurifier.Nlpurifier.remove_multiple_spaces`

    Args:
        df (str): text dataframe
        target (str): target column in the dataframe which you have to clean up

    Returns: Cleaned Data-Frame           
    """

    print_in_blue(
        f"Dataframe contains {df.shape[0]} rows and {df.shape[1]} columns\n")

    detect = _auto_detect_noise_in_textual_data()

    purifier = Nlpurifier(df, target, show_widgets=False)
    purifier.drop_null_rows()
    purifier.lower()
    purifier.remove_numbers()
    purifier.remove_html_tags()
    # purifier.remove_accented_chars()
    purifier.convert_accented_chars_to_normal()
    purifier.remove_punctuations()
    purifier.remove_stop_words()
    purifier.remove_multiple_spaces()

    if representation:
        if representation == "tfidf":
            pass
        elif representation == "tf":
            pass
        elif representation == "bow":
            pass

    print(colored("\nPurifying Completed!\n", "green", attrs=["bold"]))
    return purifier.df


class Nlpurifier:

    def __init__(self: str, df: pd.DataFrame, target: str, spacy_model="en_core_web_sm", show_widgets=True):
        self._set_df_and_target(df, target)
        self.spacy_model = spacy_model
        self.nlp = spacy.load(spacy_model)

        if show_widgets:
            self.widget = Widgets()

            self.purifier_widgets = {}
            self.word_count_series = pd.Series()
            self._show_widgets()

    def _start_purifying(self, e):

        print_in_blue(
            f"Dataframe contains {self.df.shape[0]} rows and {self.df.shape[1]} columns\n")

        if self.purifier_widgets["dropna"]:
            self.drop_null_rows()
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

        if self.purifier_widgets["remove_numbers"]:
            self.remove_numbers()
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

        if self.purifier_widgets["remove_stop_words"]:
            self.remove_stop_words()

        if self.most_common_word_range:
            word_range = self.most_common_word_range
            if self.word_count_series.empty:
                self.generate_word_count_series()
                print_in_blue(
                    "To see pandas series of word and their respective count, type '<obj>.word_count_series'\n")

            print_in_blue(f"Removing top {word_range} words from dataframe")
            word_list = self.word_count_series[:word_range].index.tolist()
            print("Which are: ", word_list)
            self.remove_words(word_list)

        if self.most_rare_word_range:
            word_range = self.most_rare_word_range
            if self.word_count_series.empty:
                self.generate_word_count_series()
                print_in_blue(
                    "To see pandas series of word and their respective count, type '<obj>.word_count_series'\n")

            print_in_blue(
                f"Removing top {word_range} rare words from dataframe")
            word_list = self.word_count_series[-word_range:].index.tolist()
            print("Which are: ", word_list)
            self.remove_words(word_list)

        if self.purifier_widgets["convert_emojis_to_word"]:
            self.convert_emojis_to_word()
        if self.purifier_widgets["convert_emoticons_to_word"]:
            self.convert_emoticons_to_word()

        if self.purifier_widgets["remove_emojis"]:
            if self.purifier_widgets["convert_emojis_to_word"]:
                print_in_red(
                    "'Remove Emojis' action is skipped because, all the emojis are converted to words.")
            else:
                self.remove_emojis()

        if self.purifier_widgets["remove_emoticons"]:
            if self.purifier_widgets["convert_emoticons_to_word"]:
                print_in_red(
                    "'Remove Emoticons' action is skipped because, all the emoticons are converted to words.")
            else:
                self.remove_emoticons()

        if self.purifier_widgets["remove_special_and_punct"]:
            self.remove_punctuations()

        if self.purifier_widgets["root_word_technique"]:
            technique = self.purifier_widgets["root_word_technique"]
            if technique == "Stemming":
                self.stemming()
            if technique == "Lemmatization":
                self.leammatize()

        print(colored("\nPurifying Completed!\n", "green", attrs=["bold"]))
        print("type <obj>.df to access processed and purified dataframe")

    def _set_root_word_technique(self, technique):
        self.purifier_widgets["root_word_technique"] = technique

    def _set_most_common_word_range(self, common_range: int) -> None:
        self.most_common_word_range = common_range
        print_in_red(f"Top {common_range} most common words will be removed")

    def _most_common_words_widget(self, condition):
        if condition:
            print("The top words maybe frequent 'Stop Words' and they will be removed instead of actual words, to avoid this consider ticking 'Remove Stop Words' checkbox")

            interact(self._set_most_common_word_range,
                     common_range=widgets.IntSlider(min=1, max=100, step=1, value=10, description="Word Range:"))
        else:
            self.most_common_word_range = None
        print("-"*50)

    def _set_most_rare_word_range(self, rare_range: int) -> None:
        self.most_rare_word_range = rare_range
        print_in_red(f"Top {rare_range} most rare words will be removed")

    def _most_rare_words_widget(self, condition):
        if condition:
            interact(self._set_most_rare_word_range,
                     rare_range=widgets.IntSlider(min=1, max=100, step=1, value=10, description="Word Range:"))
        else:
            self.most_rare_word_range = None
        print("-"*50)

    def _widget_interactions(self, dropna, lower, contraction, count_mail,
                             count_urls, word_count, remove_numbers, remove_stop_words,
                             remove_special_and_punct, remove_mail, remove_html,
                             remove_urls, remove_spaces, remove_accented,
                             convert_emojis_to_word, remove_emojis, remove_emoticons, convert_emoticons_to_word):

        self.purifier_widgets["dropna"] = True if dropna else False
        self.purifier_widgets["lower"] = True if lower else False
        self.purifier_widgets["contraction"] = True if contraction else False
        self.purifier_widgets["count_mail"] = True if count_mail else False
        self.purifier_widgets["count_urls"] = True if count_urls else False
        self.purifier_widgets["word_count"] = True if word_count else False
        self.purifier_widgets["remove_numbers"] = True if remove_numbers else False
        self.purifier_widgets["remove_stop_words"] = True if remove_stop_words else False
        self.purifier_widgets["remove_special_and_punct"] = True if remove_special_and_punct else False
        self.purifier_widgets["remove_mail"] = True if remove_mail else False
        self.purifier_widgets["remove_html"] = True if remove_html else False
        self.purifier_widgets["remove_urls"] = True if remove_urls else False
        self.purifier_widgets["remove_spaces"] = True if remove_spaces else False
        self.purifier_widgets["remove_accented"] = True if remove_accented else False
        self.purifier_widgets["convert_emojis_to_word"] = True if convert_emojis_to_word else False
        self.purifier_widgets["remove_emojis"] = True if remove_emojis else False
        self.purifier_widgets["remove_emoticons"] = True if remove_emoticons else False
        self.purifier_widgets["convert_emoticons_to_word"] = True if convert_emoticons_to_word else False

    def _show_widgets(self):

        self.dropna_widget = self.widget.checkbox(
            description='Drop Null Rows')
        self.remove_numbers_widget = self.widget.checkbox(
            description='Remove Numbers and Alphanumeric words')
        self.lower_widget = self.widget.checkbox(description='Lower all Words')
        self.contraction_widget = self.widget.checkbox(
            description='Contraction to Expansion')
        self.count_mail_widget = self.widget.checkbox(
            description='Count Mails')
        self.count_urls_widget = self.widget.checkbox(description='Count Urls')
        self.word_count_widget = self.widget.checkbox(
            description='Get Word Count')

        self.remove_emojis_widget = self.widget.checkbox(
            description='Remove Emojis')
        self.remove_emoticons_widget = self.widget.checkbox(
            description='Remove Emoticons')
        self.convert_emoticons_to_word_widget = self.widget.checkbox(
            description='Conversion of Emoticons to Words')
        self.convert_emojis_to_word_widget = self.widget.checkbox(
            description='Conversion of Emojis to Words')

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

        items = [
            [self.dropna_widget, self.lower_widget, self.contraction_widget],
            [self.count_urls_widget, self.word_count_widget, self.count_mail_widget],
            [self.remove_special_and_punct_widget, self.remove_numbers_widget,
                self.remove_stop_words_widget],
            [self.remove_accented_widget,
                self.remove_mail_widget, self.remove_html_widget],
            [self.remove_urls_widget, self.remove_spaces_widget, self.convert_emojis_to_word_widget
             ],
            [self.remove_emojis_widget, self.remove_emoticons_widget,
                self.convert_emoticons_to_word_widget]
        ]

        grid_rows = len(items)
        grid_cols = 3
        grid = GridspecLayout(grid_rows, grid_cols)
        for i in range(len(items)):
            for j in range(len(items[i])):
                grid[i, j] = items[i][j]

        self.grid_output = widgets.interactive_output(
            self._widget_interactions, {'dropna': self.dropna_widget, 'lower': self.lower_widget, 'contraction': self.contraction_widget, 'count_mail': self.count_mail_widget,
                                        'count_urls': self.count_urls_widget, 'word_count': self.word_count_widget,
                                        'remove_numbers': self.remove_numbers_widget, 'remove_stop_words': self.remove_stop_words_widget,
                                        'remove_special_and_punct': self.remove_special_and_punct_widget, 'remove_mail': self.remove_mail_widget,
                                        'remove_html': self.remove_html_widget, 'remove_urls': self.remove_urls_widget, 'remove_spaces': self.remove_spaces_widget,
                                        'remove_accented': self.remove_accented_widget, 'convert_emojis_to_word': self.convert_emojis_to_word_widget,
                                        'remove_emojis': self.remove_emojis_widget, 'remove_emoticons': self.remove_emoticons_widget, 'convert_emoticons_to_word': self.convert_emoticons_to_word_widget
                                        })

        # Displaying all methods in grid layout
        display(grid)

        # Display Root Word Widget
        print_in_blue("\nConvert Word to its Base Form")
        self.root_word_widget = widgets.RadioButtons(
            options=['None', 'Stemming', 'Lemmatization'],
            description="Technique:"
        )
        interact(self._set_root_word_technique,
                 technique=self.root_word_widget)

        # Widget for removing most common words
        print_in_red("Remove Top Common Words")
        interact(self._most_common_words_widget, condition=widgets.Checkbox(
            description="Remove Top Common Words"))

        # Widget for removing most rare words
        print_in_red("Remove Top Rare Words")
        interact(self._most_rare_words_widget, condition=widgets.Checkbox(
            description="Remove Top Rare Words"))

        # Button For Start Purifying
        start_btn = widgets.Button(description="Start Purifying")
        start_btn.on_click(self._start_purifying)
        display(start_btn)

    def _set_df_and_target(self, df, target):
        self.df = df.copy()
        if target in self.df.columns:
            self.target = target
        else:
            print_in_red(
                "Please provide correct `target` column name, containing only textual data for analysis")
            sys.exit(1)

    """
    ###############################################################################################
    Code for Nlp Purification Methods Starts
    ###############################################################################################
    """

    def get_text(self):
        self.text = " ".join(self.df[self.target])
        return self.text

    def drop_null_rows(self):
        """Drops rows having [' ', 'NULL', np.nan] values """
        total_null_rows = self.df[self.target].isin(
            [' ', 'NULL', np.nan]).sum()
        if total_null_rows > 0:
            print("Dropping rows having [' ', 'NULL', numpy.nan] values")
            self.df.dropna(inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            print_in_red(f"Total Null rows dropped: {total_null_rows}\n")
        else:
            print(colored("There is no null rows present.\n", "green"))

    @timer
    def lower(self):
        self.df[self.target] = self.df[self.target].apply(lambda x: x.lower())

    @timer
    def remove_numbers(self):
        self.df[self.target] = self.df[self.target].apply(
            lambda x: re.sub(r'[0-9]', '', x))

    @timer
    def remove_punctuations(self):
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

    def _contraction_to_expansion(self, x: str) -> str:
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
            lambda x: self._contraction_to_expansion(x))

    def _remove_accented_chars(self, x: str) -> str:
        x = unicodedata.normalize('NFKD', x).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        return x

    @timer
    def remove_accented_chars(self):
        self.df[self.target] = self.df[self.target].apply(
            lambda x: self._remove_accented_chars(x))

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
    def convert_accented_chars_to_normal(self):
        self.df[self.target] = self.df[self.target].apply(
            lambda x: unidecode.unidecode(x))

    @timer
    def get_word_count(self):
        text = self.get_text()
        text = text.split()
        self.word_count = pd.Series(text).value_counts()
        print("type <obj>.word_count for getting word count series")

    def _remove_emoji(self, x: str):
        """Removes Emoji Lambda Function
        Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b"""
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', x)

    @timer
    def remove_emojis(self):
        self.df[self.target] = self.df[self.target].apply(
            lambda x: self._remove_emoji(x))

    def _remove_emoticons(self, x: str):
        emoticon_pattern = re.compile(
            u'(' + u'|'.join(k for k in EMOTICONS) + u')')
        return emoticon_pattern.sub(r'', x)

    @timer
    def remove_emoticons(self):
        self.df[self.target] = self.df[self.target].apply(
            lambda x: self._remove_emoticons(x))

    def _convert_emoticons_to_word(self, text: str):
        for emot in EMOTICONS:
            text = re.sub(
                u'('+emot+')', "_".join(EMOTICONS[emot].replace(",", "").split()), text)
        return text

    @timer
    def convert_emoticons_to_word(self):
        self.df[self.target] = self.df[self.target].apply(
            lambda x: self._convert_emoticons_to_word(x))

    def _convert_emojis_to_word(self, text: str):
        for emot in UNICODE_EMO:
            text = re.sub(
                r'('+emot+')', "_".join(UNICODE_EMO[emot].replace(",", "").replace(":", "").split()), text)
        return text

    @timer
    def convert_emojis_to_word(self):
        self.df[self.target] = self.df[self.target].apply(
            lambda x: self._convert_emojis_to_word(x))

    def generate_word_count_series(self):
        """Generate pandas series for word counts, to access type '<obj>.word_count_series'"""
        text = " ".join(self.df[self.target])
        text = text.split()
        self.word_count_series = pd.Series(text).value_counts()

    @timer
    def remove_words(self, word_list):
        """Removes words which are in word list

        Args:
            word_list (list): List of words to be removed.
        """
        self.df[self.target] = self.df[self.target].apply(lambda x: " ".join(
            [word for word in x.split() if word not in word_list]))

    def _lemmatize(self, x: str) -> str:
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
        print(
            f"Internally for lemmatization it uses '{self.spacy_model}' spacy model,\nto change it please provide `spacy_model='your_model' in constructor`")
        self.df[self.target] = self.df[self.target].apply(
            lambda x: self._lemmatize(x))

    def _stemming(self, x):
        return " ".join([ps.stem(word) for word in x.split()])

    @timer
    def stemming(self):
        print("Using Porter Stemmer for stemming")
        self.df[self.target] = self.df[self.target].apply(
            lambda x: self._stemming(x))


if __name__ == '__main__':
    pass
