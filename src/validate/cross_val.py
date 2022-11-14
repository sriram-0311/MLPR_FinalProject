import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score , cross_validate
from sklearn.datasets import load_iris

iris_data = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)

# iris data for testing the code.
data = iris_data.data
label = iris_data.target

# Performance: accuracy , # of set: 3 
scores = cross_val_score(dt_clf , data , label , scoring='accuracy',cv=3)
print('Cross Validation Accuracy:',np.round(scores, 4))
print('Average Validation Accuracy:', np.round(np.mean(scores), 4))

