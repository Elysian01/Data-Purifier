import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from termcolor import colored


class Purify:
    def __init__(self, df):
        self.__set_df(df)

    def __set_df(self, df):
        self.df = df

    def drop_null_col(self, df, threshold) -> pd.DataFrame:
        '''
        Drop columns below having null values below certain threshold,
        and returns clean dataframe
        '''
        if threshold >= 0 and threshold <= 1:
            return df.loc[:, df.isin([' ', 'NULL', 0, np.nan]).mean() < threshold]
        raise Exception("Threshold should be between 0 and 1")


if __name__ == "__main__":
    print("Purify WOrked!!")
