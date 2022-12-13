#####################################################################################
# https://www.kaggle.com/code/sasha18/resampling-methods-using-bootstrap-cv/notebook
# https://www.analyticsvidhya.com/blog/2020/02/what-is-bootstrap-sampling-in-statistics-and-machine-learning/
#####################################################################################

# Import numerical libraries
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample  # for Bootstrap sampling
from sklearn import naive_bayes
import numpy as np
from numpy import array
import pandas as pd

# Import graphical plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Import resampling and modeling algorithms

# KFold CV

warnings.filterwarnings('ignore')

# set files path
part1 = "../../input/sign_mnist_train.csv"
part2 = "../../input/sign_mnist_test.csv"

print("*** Merging multiple csv files into a single pandas dataframe ***")

# merge files
data = pd.concat(
    map(pd.read_csv, [part1, part2]), ignore_index=True)

data.head()

values = data.values

# Lets configure Bootstrap

n_iterations = 10  # No. of bootstrap samples to be repeated (created)
# Size of sample, picking only 50% of the given data in every bootstrap sample
n_size = int(len(data) * 0.50)

# Lets run Bootstrap
stats = list()
for i in range(n_iterations):

    # prepare train & test sets
    # Sampling with replacement..whichever is not used in training data will be used in test data
    train = resample(values, n_samples=n_size)
    # picking rest of the data not considered in training sample
    test = np.array([x for x in values if x.tolist() not in train.tolist()])

    # fit model
    model = naive_bayes.GaussianNB()
    # model.fit(X_train,y_train) i.e model.fit(train set, train label as it is a classifier)
    model.fit(train[:, :-1], train[:, -1])

    # evaluate model
    predictions = model.predict(test[:, :-1])  # model.predict(X_test)
    # accuracy_score(y_test, y_pred)
    score = accuracy_score(test[:, -1], predictions)
    # caution, overall accuracy score can mislead when classes are imbalanced

    print(score)
    stats.append(score)
