# create random tree decision tree model for the dataset and report the accuracy
import numpy as np
from sklearn import tree
from sklearn import datasets
import matplotlib.pyplot as plt
import LoadImage
import os
import pickle
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV
import tensorflow_decision_forests as tfdf

# create tfdf model and train using training data
def create_tf_random_forest(tf_td):
    # create tfdf model
    model = tfdf.keras.RandomForestModel()
    # train the model
    model.fit(tf_td)
    # evaluate the model
    evaluation = model.evaluate(tf_td, return_dict=True)
    print(evaluation)
    print("Accuracy of the model is : ", evaluation['accuracy'])

# create tf gradient boosted decision tree model
def create_tf_gradient_boosted_decision_tree(X_train, y_train, X_test, y_test):
    # create tfdf model
    model = tfdf.keras.GradientBoostedTreesModel()
    # train the model
    model.fit(x=X_train, y=y_train)
    # evaluate the model
    evaluation = model.evaluate(X_test, y_test, return_dict=True)
    print(evaluation)
    print("Accuracy of the model is : ", evaluation['accuracy'])


# run the random forest model for loaded dataset from LoadImage
if __name__ == "__main__":
    # load dataset
    path = '/Users/anushsriramramesh/Downloads/archive/asl_alphabet_train/asl_alphabet_train'

    td = pd.read_csv('TrainingDataFrame.csv')
    tf_td = tfdf.keras.pd_dataframe_to_tf_dataset(td, label="Label")

    Trainingdata, TrainingdataLabel = LoadImage.load_data_tf(path)
    # split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(Trainingdata, TrainingdataLabel, test_size=0.2, random_state=42)

    # run the random forest model
    #create_tf_random_forest(X_train, y_train, X_test, y_test)
    create_tf_random_forest(tf_td)
    # # run the gradient boosted decision tree model
    # create_tf_gradient_boosted_decision_tree(X_train, y_train, X_test, y_test)