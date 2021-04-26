import pandas as pd
import numpy as np


class Mlpurifier:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def null_rows(self):
        """Drop all rows containing null values"""
        self.df.dropna().reset_index()

    def drop_null_col_based_on_percentage(self, threshold):
        '''
        Drop columns below having null values below certain threshold,
        and returns clean dataframe
        '''
        if threshold >= 0 and threshold <= 1:
            return self.df.loc[:, self.df.isin([' ', 'NULL', 0, np.nan]).mean() < threshold]
        raise Exception("Threshold should be between 0 and 1")


if __name__ == '__main__':
    ac = Mlpurifier()
