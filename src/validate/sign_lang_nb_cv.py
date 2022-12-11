import numpy as np
import pandas as pd
import timeit
from sklearn import naive_bayes
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV

# reading the csv files using pandas
train_data = pd.read_csv(
    "../../input/sign_mnist_train.csv")
test_data = pd.read_csv(
    "../../input/sign_mnist_test.csv")

# Preparing data
y_train = train_data['label']
X_train = train_data.drop(columns='label')

y_test = test_data['label']
X_test = test_data.drop(columns='label')


########################################################################
#                            Naive Baysian
# https://www.kaggle.com/code/danready/bertodaniele-mnistsign-bayes
# https://www.kaggle.com/code/marioeid/sign-language-mnist-100-accuracy
########################################################################
# GaussianNB model
start = timeit.default_timer()   
model = naive_bayes.GaussianNB()
model.fit(X_train, y_train)

# predict
y_pred_1 = model.predict(X_test)

# accuracy
print("\n")
scores = cross_val_score(model, X_train, y_train, cv=5)
stop = timeit.default_timer()
print("GaussianNB accuracy: %f" % (scores.mean() * 100))
print("time:", end - start)

# MultinomialNB model
start = timeit.default_timer()
model = naive_bayes.MultinomialNB()
model.fit(X_train, y_train)
# predict
y_pred_2 = model.predict(X_test)
# accuracy
scores = cross_val_score(model, X_train, y_train, cv=5)
stop = timeit.default_timer()
print("MultinomialNB accuracy: %f" % (scores.mean() * 100))
print("time:", end - start)

# ComplementNB model
start = timeit.default_timer()
model = naive_bayes.ComplementNB()
model.fit(X_train, y_train)
# predict
y_pred_3 = model.predict(X_test)
# accuracy
scores = cross_val_score(model, X_train, y_train, cv=5)
stop = timeit.default_timer()
print("ComplementNB accuracy: %f" % (scores.mean() * 100))
print("time:", end - start)

################################################################################
#                           Logistic Regression
# https://www.kaggle.com/code/heraclex12/hog-and-logistic-regression-achieved-87
################################################################################


def read_file(filename, hist=False):
    """
      Read csv file.
      if hist=True, images will be load in Image Histogram, a vector has 256 elements
    """
    df = pd.read_csv(filename)
    if hist:
        pixels = np.empty((df.shape[0], 256))
        for i in range(df.shape[0]):
            tmp = df.iloc[i].value_counts(sort=False).reindex(
                range(0, 256), fill_value=0).to_numpy()
            pixels[i] = tmp
    else:
        pixels = df.drop(columns=['label']).values

    labels = df.label.values
    pixels = pixels.astype(np.float64)
    return pixels, labels


# Read image histogram
X_train_hist, y_train_hist = read_file(
    "../../input/sign_mnist_train.csv", hist=True)

# Train model with histogram features
start = timeit.default_timer()
histogram_model = LogisticRegression(
    max_iter=10000, solver='lbfgs', multi_class='auto')
scores = cross_val_score(histogram_model, X_train_hist, y_train_hist, cv=5)
stop = timeit.default_timer()
print("Logistic-regression accuracy: %f" % (scores.mean() * 100))
print("time:", end - start)
