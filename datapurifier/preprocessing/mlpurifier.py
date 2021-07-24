import pandas as pd
import numpy as np
from termcolor import colored

from IPython.display import display



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
    
    def drop_single_value_column(self):
        """Drop columns having a single value or zero variance
        """
        pass
        
    def drop_duplicate_rows(self):
        """Drop Duplicate Rows from dataset
        """
        pass
    
    
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


if __name__ == '__main__':
    ac = Mlpurifier()
