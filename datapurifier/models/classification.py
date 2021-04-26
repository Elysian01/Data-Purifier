
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # SVR for regression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


class Classification:
    def __init__(self, X_train, y_train, X_test, y_test, algorithms: list):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.algorithms = algorithms

    def get_models(self):
        self.classification_models = {
            "logistic": LogisticRegression,
            "decision": DecisionTreeClassifier,
            "random": RandomForestClassifier,
            "knn": KNeighborsClassifier,
            "gaussian-nb": GaussianNB,
            "multinomial-nb": MultinomialNB,
            "bernoulli-nb": BernoulliNB
        }
        return self.classification_models
