import os
import re
import pandas as pd
from urllib.request import urlopen, urlretrieve

# Code is referred from seaborn library
# https://github.com/mwaskom/seaborn/blob/master/seaborn/utils.py


def get_text_dataset_names() -> list:
    """Gets names of all textual datasets for nlp"""
    datasets = ["womens_clothing_e-commerce_reviews"]
    return datasets


def get_dataset_names() -> list:
    """To get Dataset name
    Requires an internet connection.
    """
    url = "https://github.com/Elysian01/Data-Purifier-Dataset"
    with urlopen(url) as resp:
        html = resp.read()

    pat = r"/Elysian01/Data-Purifier-Dataset/blob/master/(\w*).csv"
    datasets = re.findall(pat, html.decode())
    return datasets


def get_data_home(data_home=None):
    """Return a path to the cache directory for example datasets.
    This directory is then used by :func:`load_dataset`.
    If the ``data_home`` argument is not specified, it tries to read from the
    ``DATAPURIFIER_DATASET`` environment variable and defaults to ``~/datapurifier-dataset``.
    """
    if data_home is None:
        data_home = os.environ.get('DATAPURIFIER_DATASET',
                                   os.path.join('~', 'datapurifier-dataset'))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def load_dataset(name, cache=True, data_home=None, **kws):
    """Load an example dataset from the online repository (requires internet).
    Use :func:`get_dataset_names` to see a list of available datasets.
    Parameters
    ----------
    name : str
        Name of the dataset (``{name}.csv`` on
        https://github.com/Elysian01/Data-Purifier-Dataset).
    cache : boolean, optional
        If True, try to load from the local cache first, and save to the cache
        if a download is required.
    data_home : string, optional
        The directory in which to cache data; see :func:`get_data_home`.
    kws : keys and values, optional
        Additional keyword arguments are passed to passed through to
        :func:`pandas.read_csv`.
    Returns
    -------
    df : :class:`pandas.DataFrame`
        Tabular data
    """
    path = ("https://raw.githubusercontent.com/"
            "Elysian01/Data-Purifier-Dataset/master/{}.csv")
    full_path = path.format(name)

    if cache:
        cache_path = os.path.join(get_data_home(data_home),
                                  os.path.basename(full_path))
        if not os.path.exists(cache_path):
            if name not in get_dataset_names():
                raise ValueError(
                    f"'{name}' is not one of the example datasets.")
            urlretrieve(full_path, cache_path)
        full_path = cache_path

    df = pd.read_csv(full_path, **kws)

    if df.iloc[-1].isnull().all():
        df = df.iloc[:-1]

    return df
