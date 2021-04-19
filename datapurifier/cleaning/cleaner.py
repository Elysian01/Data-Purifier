import pandas as pd
import numpy as np


class AutoCleaner:
    def drop_null_col(self, df, threshold):
        '''
        Drop columns below having null values below certain threshold,
        and returns clean dataframe
        '''
        if threshold >= 0 and threshold <= 1:
            return df.loc[:, df.isin([' ', 'NULL', 0, np.nan]).mean() < threshold]
        raise Exception("Threshold should be between 0 and 1")


if __name__ == '__main__':
    ac = AutoCleaner()
