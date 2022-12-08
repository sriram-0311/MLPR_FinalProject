#preprocessing functions for our image data
import cv2
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import datasets
import matplotlib.pyplot as plt
import LoadImage
import os
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV

# draw decision tree for training data
def draw_decision_tree(X, y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    tree.plot_tree(clf)
    #plt.show()
    # save decision tree as pdf
    tree.export_graphviz(clf, out_file='tree.dot')
    # convert dot file to pdf
    os.system('dot -Tpdf tree.dot -o tree.pdf')
    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    print(" number of nodes in tree : ", clf.tree_.node_count)
    print(" number of nodes in tree : ", clf.tree_.max_depth)

# load model and predict
def load_model_predict(X_train, y_train,X_test):
    # load the model from disk
    loaded_model = pickle.load(open('modelwithmaxdepth4.sav', 'rb'))
    loaded_model.fit(X_train, y_train)
    result = loaded_model.predict(X_test)
    print(" number of nodes in tree : ", loaded_model.tree_.node_count)
    print(" number of nodes in tree : ", loaded_model.tree_.max_depth)
    return result

# prune the tree using cross validation score
def prune_tree(X, y):
    # create a list of values to try for max_depth:
    max_depth_range = list(range(1, 5))
    # list to store the average RMSE for each value of max_depth:
    accuracy = []
    for depth in max_depth_range:
        print("depth : ", depth)
        clf = tree.DecisionTreeClassifier(max_depth=depth, random_state=1)
        scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
        pickle.dump(clf, open('modelwithmaxdepth' + str(depth) + '.sav', 'wb'))
        accuracy.append(scores.mean())
    plt.plot(max_depth_range, accuracy)
    plt.xlabel('max_depth')
    plt.ylabel('accuracy')
    plt.show()

# calculate accuracy of the model
def calculate_accuracy(y_test, y_pred):
    count = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            count += 1
    print("Accuracy of the model is : ", count / len(y_test))
# def regularization()

# def PCA()

# def threshold_image(image):
#     grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

#     # global thresholding
#     ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#     # Otsu's thresholding
#     ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     # Otsu's thresholding after Gaussian filtering
#     blur = cv2.GaussianBlur(img,(5,5),0)
#     ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#     return th3

def blob_detection(im):
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector()

    # Detect blobs.
    keypoints = detector.detect(im)
    return keypoints

def plot_image(img):
    cv2.imshow('Black white image', img)
    # cv2.imshow('Original image',originalImage)
    # cv2.imshow('Gray image', grayImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Read image from csv file
    # df = read_image_from_csv('TrainingdataFrame.csv')
    # load data from path
    path = '/Users/anushsriramramesh/Downloads/archive/asl_alphabet_train/asl_alphabet_train'
    Trainingdata, TrainingdataLabel = LoadImage.load_data_tf(path)
    # split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(Trainingdata, TrainingdataLabel, test_size=0.2, random_state=42)
    # decision tree
    #draw_decision_tree(X_train, y_train)

    # prune tree
    #prune_tree(X_train, y_train)

    # load model and predict
    y_pred = load_model_predict(X_train,y_train,X_test)
    #calculate accuracy
    calculate_accuracy(y_test, y_pred)
    # load pruned tree model
    load_pruned_tree_model()

    # Show a random image
    # img = df['Image'][np.random.randint(0,len(df['Image']))]
    # img = threshold_image(img)
    # keypoints = blob_detection(img)
    # print(keypoints)
    # plot_image(img)