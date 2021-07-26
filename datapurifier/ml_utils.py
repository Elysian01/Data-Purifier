import numpy as np
import pandas as pd
import seaborn as sns
from termcolor import colored
import matplotlib.pyplot as plt
from IPython.display import display

import warnings
warnings.filterwarnings("ignore")

def get_categorical_columns(df) -> list:
    '''Return list of categorical features
    '''
    
    cat_cols = df.select_dtypes(include="O").columns.tolist()
    return cat_cols

def get_numerical_columns(df) -> list:
    '''Return list of numerical features
    '''
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_cols = df.select_dtypes(include=numerics).columns.tolist()
    return num_cols

def shape(df, formatted_output: bool = False) -> None:
    if not formatted_output:
        print("Shape of DataFrame: ", df.shape)
        return
    print(
        colored(f"Dataframe contains {df.shape[0]} rows and {df.shape[1]} columns", "blue", attrs=["bold"]))

def sample(df, num_or_rows: int = 10, all_columns: bool = True) -> None:
    if all_columns:
        pd.set_option("display.max_columns", None)
    print(colored("\nSample of Dataframe:", "red", attrs=["bold"]))
    display(df.sample(num_or_rows))



def columns_type(df, verbose=False) -> tuple:
    '''return number of categorical and numerical columns
    '''
    
    cat_columns = get_categorical_columns(df)
    num_columns = get_numerical_columns(df)
    
    if verbose:
        print(colored(
            f"\nCategorical columns: ", "green", attrs=["bold"]), end="")
        print(cat_columns)
        print(colored(
            f"\nNumerical columns: ", "green", attrs=["bold"]), end="")
        print(num_columns)
         
    return (len(cat_columns), len(num_columns))

def null_columns_percentage(df) -> pd.DataFrame:
    '''Prints Null Information of dataframe, i.e. only the number of rows having null values and their null percentage
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

def unique_df_percentage(df) -> pd.DataFrame:
    """Returns dataframe containing (column name, count of total unique values per column, uniqueness percentage) 
    """
    
    data = {"column": [], "unique_count": [], "unique_percent": []}
    for col in df.columns:
        data["column"].append(col)
        data["unique_count"].append(df[col].nunique()) 
        data["unique_percent"].append(data["unique_count"][-1]/ df.shape[0] *100)
        
    unique_df = pd.DataFrame(data)
        
    return unique_df

"""##################################################################################################################################
Drop functions
##################################################################################################################################"""


def drop_single_value_column(df):
    """Drop columns having a single value or zero variance
    """
    counts = df.nunique()
    # record columns to delete
    to_del = [i for i,val in enumerate(counts) if val == 1]
    if to_del:
        print("Dropped columns containing single value: ", to_del)
        df.drop(to_del, axis=1, inplace=True)
    else:
        print("No columns contains single value")
    return df   

def drop_column_based_on_uniqueness_threshold(df: pd.DataFrame, uniqueness_threshold: float = 0.95, exact: bool = False) -> pd.DataFrame:
    '''Removes columns having unique value more than 95% (uniqueness_threshold), like: ID, serial_no, etc
    
    Parameters:
        df (pd.DataFrame): input dataframe
        uniqueness_threshold (float): value of uniqueness column precentage is between 0 and 100
        exact (bool): deletes column which have exactly 'unique column precentage' equal to 'uniqueness_threshold'

    Returns:
        Pandas dataframe with columns deleted having unique value more than or equal to 'uniqueness_threshold'
    '''
    
    to_del = []
    for col in df.columns:
        if exact:
            if df[col].nunique()/df.shape[0] == uniqueness_threshold:
                to_del.append(col)
        else:
            if df[col].nunique()/df.shape[0] >= uniqueness_threshold:
                to_del.append(col)

    if to_del:
        if exact:
            print(colored(
            f"\nDroping columns having uniqueness_threshold={uniqueness_threshold}\n", "red", attrs=["bold"]))
        else:
            print(colored(
                f"\nDroping columns having uniqueness_threshold={uniqueness_threshold} or greater'\n", "red", attrs=["bold"]))
        df.drop(to_del, axis=1, inplace=True)
        print("Dropped Columns are: ", to_del)
    else:
        if exact:
            print(f"No columns found having uniqueness_threshold of={uniqueness_threshold} or greater")
        else:
            print(f"No columns found having uniqueness_threshold of={uniqueness_threshold}")
    return df
    
def drop_null_rows(df):
    """Drop all rows containing null values"""
    return df.dropna().reset_index(drop=True)
    

def drop_null_col_based_on_percentage(df, threshold):
    '''Drop columns below having null values below certain threshold,
    and returns clean dataframe
    '''
    if threshold >= 0 and threshold <= 1:
        return df.loc[:, df.isin([' ', 'NULL', 0, np.nan]).mean() < threshold]
    raise Exception("Threshold should be between 0 and 1")


def drop_duplicate_rows(df):
    """Drop Duplicate Rows from dataset
    """
    print("Checking for duplicate rows...")
    if df.duplicated().sum() == 0:
        print("Dataframe contains no duplicate rows")
    else:
        initial_rows = df.shape[0]
        df.drop_duplicates(inplace=True)
        print(colored(f"Dropped {initial_rows - df.shape[0]} duplicate rows", "red", attrs=["bold"]))
    return df
