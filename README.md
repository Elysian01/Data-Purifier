# Data-Purifier

### A Python library for Automated Exploratory Data Analysis, Automated Data Cleaning and Automated Data Preprocessing For Machine Learning and Natural Language Processing in Python.


## Features

* It gives shape, number of categorical and numerical features, description of the dataset, and also the information about the number of null values and their respective percentage. 

* For understanding the distribution of datasets and getting useful insights, there are many interactive plots generated where the user can select his desired column and the system will automatically plot it. Plot includes
   1. Count plot
   2. Correlation plot
   3. Joint plot
   4. Pair plot
   5. Pie plot 


Demo Output of Auto EDA
<br><br>
<img src = "./static/demo.gif" width="600px" height = "300px">


## Get Started

Install the package
```
pip install data-purifier

python -m spacy download en_core_web_sm
```

Load the module
```python
from datapurifier import Mleda
```

Load the dataset and let the magic of automated EDA begin

```python
df = pd.read_csv("./datasets/iris.csv")
ae = MLeda(df)
ae
```

Example: https://colab.research.google.com/drive/1J932G1uzqxUHCMwk2gtbuMQohYZsze8U?usp=sharing

Python Package: https://pypi.org/project/data-purifier/