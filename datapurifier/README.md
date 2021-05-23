<!---
## List of Functions and Class In Auto EDA.
-->

### An Automated Exploratory Data Analysis Library For Machine Learning and Deep Learning in Python.

## Steps in Auto EDA

- [x] 1. **Shape** of dataset
- [x] 2. **Sample** of dataset
- [x] 3. **Number of categorical and numerical** attributes
- [x] 4. **Null values** in the dataset (number & %) => recommend to drop higher %
- [x] 5. **Value count** of attribute (unique) in the dataset
- [ ] 6. **Describe** the data (5 point summary) => inference (min, max, dist) => recommend columns to watch out for => outliers.
- [ ] 7. **Distribution** of dataset
        a. Skewed or Normal
- [ ] 8. Perform **grouping/aggregation** wherever necessary 
- [ ] 9. **Recommend Columns** or attribute to remove
        a. Column with all value as unique (e.g: ID)
- [ ] 10. Give **insight from the data** and suggest ml algorithms, techniques, cleaning, etc hints, tips and tricks
- [x] 11. **Correlation** matrix (Heatmap)
- [ ] 12. **Distribution Plot**
- [x] 13. Joint Plot 
- [x] 14. Pair plot
- [x] 15. Pie Chart
- [ ] 16. Box or Violen plot

## Steps in NLP EDA

- Summarize main characteristics of data.

- [x] EDA for NLP (https://towardsdatascience.com/exploratory-data-analysis-for-natural-language-processing-ff0046ab3571)
- [x] Model For Sentiment Analysis
- [x] Word Frequency Analysis
  - [x] Total words which occur one, two or 3 times only
- [x] Top Word (Visulize it via word cloud)
- [x] Given the word, finds it count

## Cleaning

[notebook reference](https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing#Introduction)
- [x] Removing Emoji
- [x] Removal of emoticons
- [x] Conversion of emoticons to words
- [x] Conversion of emojis to words
- [x] Radio-button for lemmatization and stemming
- [x] Removal of Frequent words
- [x] Removal of Rare words

- [x] Add inbuild dataset functionality and its access.
- [x] Documentation for inbuild dataset, version, cleaning methods functionality
- [x] dp.__version__ functionality

## Analysis
- [ ] LDA
- [ ] PCA (Principal Component Analysis)

## Machine Learning and Deep Learning Model input preprocessing.

- [ ] Word 2 Vector
  - [ ] Count
  - [ ] TF-IDF
  - [ ] Hash
  - [ ] Word Embedding
- [ ] Encoding categorical values

## Automated Visualization

- [ ] Perform Automated Data Visualization

- [ ] Report Generation of Executed Process.

## Automated Data Cleaning
- [ ] Automated Data Cleaning
  - [ ] Null/Missing values handling
  - [ ] Encoding Categorical values
  - [ ] Splitting dataset into training and test set
  - [ ] Feature scaling

### Cleaning Reference
- [javatpoint](https://www.javatpoint.com/data-preprocessing-machine-learning)

## Model Evaluation
- [ ] Classification Evaluation
  - [ ] Classification Report
  - [ ] Confusion Matrix
- [ ] Regression Evaluation
  - [ ] MSE, MAE, RMSE

## Reference

[Towards Data science blog](https://towardsdatascience.com/nlp-part-3-exploratory-data-analysis-of-text-data-1caa8ab3f79d)

[Vidhya Analysis blog](https://www.analyticsvidhya.com/blog/2020/04/beginners-guide-exploratory-data-analysis-text-data/)

[Medium](https://medium.com/analytics-vidhya/how-to-begin-performing-eda-on-nlp-ffdef92bedf6)

[Kaggle](https://www.kaggle.com/wil2210/eda-nlp-ml)


## List of Functions and Class In Data Purifier

## Steps

- [ ] Machine Learning
  
  - [ ] Data Distribution
    - [ ] Gaussian
    - [ ] Probability
    - [ ] Binomial 
    - [ ] Poisson

  - [ ] Missing Value Techniques
      - [ ] Deleting Row
      - [ ] Deleting Column
      - [ ] Missing Value Imputation
      - [ ] Numerical Value Imputation
      - [ ] IterativeImputer
  
- [ ] Deep Learning
  
- [ ] Natural Language Processing 
  - [x]  Text Preprocessing
    - [x]  Stop words removal
    - [x]  punctuation removal
    - [x]  Quotes word to normal word conversation
    - [x]  Number removal
  - [ ] Emoji removal
  - [x] HTML left-outs removal 
  
- [ ] Suggest Data Cleaning Techniques and methods

## Final Goals

- [ ] Automated Data Cleaning
- [ ] Automated Data Preprocessing
- [ ] Test on atleast 50 dataset before publishing.
- [ ] Display the outputs using `termcolor`
- [ ] Release the beta version to 15 testers
- [ ] Test the library for jupyter and spyder IDE
- [ ] Performing all operations/steps column-wise

        

## Points to Remember & Small Tasks

- [ ] Always write and keep track of inferences from all the steps.
- [ ] Divide into functions and classes
- [ ] Plots should be interactive (plotly)
- [ ] Params to Functions
  - [ ] Can choose to show outputs/data while performing EDA 
  - [ ] Display all columns (True or False)


## TODO

- [ ] Summary of EDA (Append all the inference of each step) => autoeda.summary()
- [ ] Test on atleast 50 dataset before publishing.
- [ ] Display the outputs using `termcolor`
- [ ] Release the beta version to 15 testers
- [ ] Test the library for jupyter and spyder IDE
- [ ] Performing all operations/steps column-wise
- [ ] Instead of performing all EDA, option to perform `basic` EDA (given as parameter to constructor)
- [ ] user can access special class variables which are set after analysis (like mleda.cat_columns)

## Deployment Testing

- [x] Jupyter Notebook
- [x] Kaggle
- [ ] Colab
- [ ] Binder