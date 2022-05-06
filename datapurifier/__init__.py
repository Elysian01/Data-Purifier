"""An Automated Exploratory Data Analysis Library For Machine Learning and Deep Learning in Python.

Classes:
    MLeda
    Nlpeda
    
Methods:
    pass
    
"""


from datapurifier.eda.mleda import Mleda
from datapurifier.eda.nlpeda import Nlpeda
from datapurifier.preprocessing.mlpurifier import Mlpurifier
from datapurifier.preprocessing.nlpurifier import Nlpurifier
from datapurifier.preprocessing.nlpurifier import NLAutoPurifier

from .dataset import *
from .report import MlReport

__version__ = "0.3.5"

# from datapurifier.preprocessing.contractions import CONTRACTIONS

# from .main import Purify
