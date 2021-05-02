import os
import pandas as pd

DATASET_DIR = os.path.abspath("datasets/")
# ALl Dataset


def get_dataset_names() -> list:
    """Gets names of all datasets"""

    datasets = [os.path.splitext(dataset)[0]
                for dataset in os.listdir(DATASET_DIR)]
    return datasets


def get_text_dataset_names() -> list:
    """Gets names of all textual datasets for nlp"""
    datasets = ["womens_clothing_e-commerce_reviews"]
    return datasets


def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load the dataset from the dataset list provided by dp.get_dataset_names() function

    Arguments:
    dataset_name: name of the dataset

    Return:
    Pandas Dataframe of the dataset
    """

    try:
        dataset = dataset_name + ".csv"
        dataset = os.path.join(DATASET_DIR, dataset)
        return pd.read_csv(dataset)
    except FileNotFoundError:
        raise Exception(
            "Please provide correct dataset name\nFor seeing all dataset available execute 'dp.get_dataset_names()' and then pass one of the dataset names as parameter to function")
