"""Machine Learning EDA

Performs Automatic Exploratory Data Analysis for Machine Learning datasets.

Class:
    MLeda
    
Methods:
    
Variables:
    cat_cols
    num_cols
"""

from datapurifier.widgets import Widgets
# from ipywidgets.widgets import widget
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from termcolor import colored
from IPython.display import display
import ipywidgets as widgets
import ipywidgets
import warnings
warnings.filterwarnings("ignore")


# %matplotlib inline
# sns.set_theme(style="darkgrid")


class Mleda:
    """
    Performs Automatic Exploratory Data Analysis for ML datasets.

    Parameters
    ----------
    df : Pandas Dataframe
    """

    def __init__(self, df: pd.DataFrame):
        self._set_df(df)
        self._init_variable(df)

        self.inference_summary = []
        self.widget = Widgets()

        self.shape(df)
        self.sample(df)
        # results = self._remove_unique_columns(df)
        # if results[0]:
        #     df = df.drop(columns=results[1])
        self.statistics(df)
        self.visualize(df)

    def _set_df(self, df):
        self.df = df

    def _init_variable(self, df):
        self.cat_cols = self.get_categorical_columns(df)
        self.num_cols = self.get_numerical_columns(df)

    def get_categorical_columns(self, df) -> list:
        '''
        Return list of categorical features
        '''
        self.cat_cols = df.select_dtypes(include="O").columns.tolist()
        return self.cat_cols

    def get_numerical_columns(self, df) -> list:
        '''
        Return list of numerical features
        '''
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        self.num_cols = df.select_dtypes(include=numerics).columns.tolist()
        return self.num_cols

    def shape(self, df, formatted_output: bool = False) -> None:
        if not formatted_output:
            print("Shape of DataFrame: ", df.shape)
            return
        print(
            colored(f"Dataframe contains {df.shape[0]} rows and {df.shape[1]} columns", "blue", attrs=["bold"]))

    def sample(self, df, num_or_rows: int = 10, all_columns: bool = True) -> None:
        if all_columns:
            pd.set_option("display.max_columns", None)
        print(colored("\nSample of Dataframe:", "red", attrs=["bold"]))
        display(df.sample(num_or_rows))

    def _remove_unique_columns(self, df, uniqueness_threshold: float = 0.8) -> list:
        '''
        Removes columns having unique value more than 80% (uniqueness_threshold), like: ID, serial_no, etc
        '''
        unique_val_cols = []
        for col in df.columns:
            if df[col].nunique()/df.shape[0] > uniqueness_threshold:
                unique_val_cols.append(col)

        if unique_val_cols:
            print(colored(
                f"\nDroping columns having number of unique value more than {uniqueness_threshold*100}%\n", "red", attrs=["bold"]))
            print("Droped Columns are: ", unique_val_cols)
            print("-"*50, "\n")
            return [True, unique_val_cols]
        return [False, []]

    def columns_type(self, df) -> tuple:
        '''
        return number of categorical and numerical columns
        '''
        cat_columns = len(self.cat_cols)
        num_columns = len(self.num_cols)
        return (cat_columns, num_columns)

    def null_columns_percentage(self, df) -> pd.DataFrame:
        '''
        Prints Null Information of dataframe,i.e. only the number of rows having null values and their null percentage
        '''
        print("\nNull Information of Dataframe: \n")
        null_df = pd.DataFrame(df.isnull().sum()).reset_index()
        null_df.columns = ["column_name", "null_rows"]
        null_df["null_percentage"] = null_df["null_rows"]*100 / df.shape[0]
        null_df = null_df[null_df["null_percentage"] != 0].sort_values(
            "null_percentage", ascending=False).reset_index(drop=True)
        print(colored(
            f"\nThere are total {null_df.shape[0]} columns having null values out of {df.shape[1]} columns in dataframe\n", "red", attrs=["bold"]))
        display(null_df)
        return null_df

    def statistics(self, df) -> None:
        '''
        Gives number of categorical and numerical columns, description and info regarding the dataframe
        '''
        attribute_types = self.columns_type(df)
        print(colored(
            f"\nThere are total {attribute_types[0]} categorical and {attribute_types[1]} numerical columns\n", "blue", attrs=["bold"]))

        print(colored("Description of Data:\n", "red", attrs=["bold"]))
        display(df.describe())

        print(colored("Information regarding data: \n", "red", attrs=["bold"]))
        display(df.info())

        if df.isnull().sum().sum() > 0:
            self.null_columns_percentage(df)
        else:
            print(colored(
                "\nCongrats!!, The Dataframe has NO NULL VALUES\n", "green", attrs=["bold"]))

    def _value_counts_plot(self, column, n: int) -> None:

        selected_columns = self.df[column].value_counts().index.tolist()[:n]
        selected_columns_count = self.df[column].value_counts().tolist()[:n]

        plt.figure(figsize=(10, 5))
        sns.barplot(x=selected_columns,
                    y=selected_columns_count)
        plt.xticks(rotation=75)
        plt.ylabel("count")
        plt.xlabel(column)

        value_count_df = pd.DataFrame()
        value_count_df["column_value"] = selected_columns
        value_count_df["count"] = selected_columns_count
        value_count_df["count_percentage"] = np.array(
            selected_columns_count) * 100 / self.df.shape[0]
        value_count_df.reset_index(drop=True, inplace=True)
        display(value_count_df)
        # print("\nTop Columns with most common values are: \n")
        # print(self.df[column].value_counts().index.tolist()[:n])

    def value_counts(self, df) -> None:
        '''
        Show Value count plots
        '''
        self._set_df(df)
        print(colored("\nInteractive Value Count Plot:\n",
                      "red", attrs=["bold"]))
        columns = df.columns
        column_dropdown = self.widget.dropdown(columns, columns[0], "Columns:")

        top_columns = self.widget.int_slider(
            minimum=1, maximum=35, step=1, value=5, description="Top Columns: ")

        items = [column_dropdown, top_columns]
        value_count_ui = widgets.HBox(items)

        out = widgets.interactive_output(
            self._value_counts_plot, {'column': column_dropdown, 'n': top_columns})

        display(value_count_ui, out)

    def _pie_plot(self, column: str) -> None:
        plt.figure(figsize=(10, 5))
        self.df[column].value_counts().plot.pie(
            autopct="%1.1f%%")

    def pie(self, df) -> None:

        cat_columns = self.get_categorical_columns(df)

        if cat_columns:
            self._set_df(df)
            print(colored("\nPie Plot:\n", "red", attrs=["bold"]))

            cat_column_dropdown = self.widget.dropdown(
                cat_columns, cat_columns[0], "Columns:")

            ipywidgets.interact(self._pie_plot, column=cat_column_dropdown)

    def _plot_joinplot(self, x: str, y: str, kind: str, hue: str) -> None:
        try:
            if hue:
                sns.jointplot(x=x, y=y, kind=kind, hue=hue, data=self.df)
            else:
                sns.jointplot(x=x, y=y, kind=kind, data=self.df)
        except Exception as e:
            print(colored(f"Error: {e}", "red", attrs=["bold"]))

    def jointplot(self, df) -> None:
        self._set_df(df)
        print(colored("\nJoint Plot:\n", "red", attrs=["bold"]))

        col = df.columns
        column_dropdown1 = self.widget.dropdown(
            col, col[0], "X-axis:")

        column_dropdown2 = self.widget.dropdown(
            col, col[0], "Y-axis:")

        kind = ["scatter", "kde", "hist", "hex", "reg", "resid"]

        kind_dropdown = self.widget.dropdown(
            kind, kind[0], "Kind:")

        hue_dropdown = self.widget.dropdown(
            col, None, "Hue:")

        items = [column_dropdown1, column_dropdown2,
                 kind_dropdown, hue_dropdown]

        joint_plot_ui = widgets.HBox(items)
        output = widgets.interactive_output(self._plot_joinplot, {
                                            'x': column_dropdown1, 'y': column_dropdown2, 'kind': kind_dropdown, "hue": hue_dropdown})

        display(joint_plot_ui, output)

    def corr(self, df) -> None:
        print(colored("\nCorrelation Heatmap Plot:\n", "red", attrs=["bold"]))
        sns.heatmap(df.corr(), annot=True, linewidth=3)
        plt.show()

    def _plot_pairplot(self, hue: str, plot_status: bool) -> None:
        if plot_status == True:
            if hue:
                sns.pairplot(self.df, hue=hue)
            else:
                sns.pairplot(self.df)
        print("-"*50)

    def pairplot(self, df) -> None:
        print(colored("Pair Plot:\n", "red", attrs=["bold"]))

        col = self.get_categorical_columns(df)
        hue_dropdown = self.widget.dropdown(
            options=col, value=None, description="Select Hue: ")

        show_pairplot = self.widget.checkbox(description='Show Pair Plot')

        items = [hue_dropdown, show_pairplot]
        pair_plot_ui = widgets.VBox(items)
        output = widgets.interactive_output(self._plot_pairplot, {
                                            'hue': hue_dropdown, 'plot_status': show_pairplot})

        display(pair_plot_ui, output)

    def visualize(self, df) -> None:
        self._set_df(df)
        self.value_counts(df)
        self.jointplot(df)
        self.corr(df)
        self.pairplot(df)
        self.pie(df)


if __name__ == "__main__":
    df = pd.read_csv("../datasets/SampleSuperstore.csv")
    ae = Mleda(df)
    print(ae.cat_columns)
