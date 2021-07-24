import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display

from termcolor import colored

import warnings
warnings.filterwarnings("ignore")

def get_categorical_columns(df) -> list:
        '''
        Return list of categorical features
        '''
        cat_cols = df.select_dtypes(include="O").columns.tolist()
        return cat_cols

def get_numerical_columns(df) -> list:
    '''
    Return list of numerical features
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
    '''
    return number of categorical and numerical columns
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

def null_columns_percentage( df) -> pd.DataFrame:
    '''
    Prints Null Information of dataframe, i.e. only the number of rows having null values and their null percentage
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