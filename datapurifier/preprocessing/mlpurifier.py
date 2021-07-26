import pandas as pd
import numpy as np
from termcolor import colored

from IPython.display import display

from datapurifier import ml_utils



class Mlpurifier:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
        # Drop Single value columns 
        self.df = ml_utils.drop_single_value_column(self.df)
        print()
        
        # Drop duplicate rows
        self.df = ml_utils.drop_duplicate_rows(self.df)
        print()



if __name__ == '__main__':
    # mlp = Mlpurifier(df)
    pass
