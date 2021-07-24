import pandas as pd
from termcolor import colored
from IPython.display import display

from datapurifier import ml_utils


# Given dataset and target variable as input following report will be generated
report_details = [
    "10 random rows of dataset",
    "Shape of dataset",
    "Total numerical and categorical column",
    "Percentage distribution of target column",
    "Description of dataset",
    "Missing Value information",
    "Possible Outlier Information",
]

class MlReport:
    
    def __init__(self, df: pd.DataFrame, target_column:str ):
        
        self.df = df
        self.target_column = target_column
        
        # Show 10 random rows of dataset
        ml_utils.sample(df)
        
        # Shape of dataset
        ml_utils.shape(df)
        
        # Total numerical and categorical column
        attribute_types = ml_utils.columns_type(df, verbose=True)
        print(colored(
            f"\nThere are total {attribute_types[0]} categorical and {attribute_types[1]} numerical columns\n", "blue", attrs=["bold"]))
        
        # Percentage distribution of target column
        
        # Description of dataset
        print(colored("Description of Data:\n", "red", attrs=["bold"]))
        display(df.describe())
        
        # Missing Value information
        if df.isnull().sum().sum() > 0:
            ml_utils.null_columns_percentage(df)
        else:
            print(colored(
                "\nCongrats!!, The Dataframe has NO NULL VALUES\n", "green", attrs=["bold"]))
        
        # Possible Outlier Information
        
        
        