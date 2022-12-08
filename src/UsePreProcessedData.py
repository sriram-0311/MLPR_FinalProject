# use /preprocessed_as_train.csv data for training decision forest models

import numpy as np
from sklearn import tree
from sklearn import datasets
import matplotlib.pyplot as plt
import LoadImage
import os
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn import tree

# load csv file and check the first 5 rows
df = pd.read_csv('preprocessed_asl_train_reformatted.csv')

# print the number of columns in the dataset
#print(df.shape)

# run sklearn decision forest models on the dataset
X = df.iloc[:, 3:63]
y = df.iloc[:, 63]

# print the shape of X and y
print(X.shape)
print(y.shape)

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train random forest model on training set
rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rf.fit(X_train, y_train)

# draw the random forest model
tree.plot_tree(rf.estimators_[0], filled=True)
plt.savefig('rf_tree.png', dpi=700)

# train decision tree model on training set
dt = tree.DecisionTreeClassifier()
dt.fit(X_train, y_train)

# evaluate the decision tree model on the test set
dt_score = dt.score(X_test, y_test)
print("Decision Tree Accuracy: ", dt_score)

# draw the decision tree model
tree.plot_tree(dt, filled=True)
plt.savefig('dt_tree.png', dpi=700)

# print the number of nodes in the decision tree
print(" number of nodes in tree : ", dt.tree_.node_count)
print(" number of nodes in tree : ", dt.tree_.max_depth)


# train adaboost model on training set
ada = AdaBoostClassifier(n_estimators=100, random_state=0)
ada.fit(X_train, y_train)

# evaluate the random forest model on the test set
rf_score = rf.score(X_test, y_test)
print("Random Forest Accuracy: ", rf_score)

# evaluate the adaboost model on the test set
ada_score = ada.score(X_test, y_test)
print("AdaBoost Accuracy: ", ada_score)




