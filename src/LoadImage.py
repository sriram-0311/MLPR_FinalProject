import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

#print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))

# Load training and test data from given path
def load_data(path):
    TrainingdataFrame = pd.DataFrame(columns=['Image', 'Label'])
    TestdataFrame = pd.DataFrame(columns=['Image'])

    TrainingPath = path + '/asl_alphabet_train/asl_alphabet_train/'
    TestPath = path + '/asl_alphabet_test/asl_alphabet_test/'

    NumberOfClasses = os.listdir(TrainingPath)
    NumberOfClasses.sort()
    if '.DS_Store' in NumberOfClasses:
        NumberOfClasses.remove('.DS_Store')
    #print(NumberOfClasses)

    for i in range(len(NumberOfClasses)):
        ClassPath = TrainingPath + NumberOfClasses[i] + '/'
        Images = os.listdir(ClassPath)
        #print("images in class ", NumberOfClasses[i], " are ", Images)
        for j in range(len(Images)):
            img = cv2.imread(TrainingPath + NumberOfClasses[i] + '/' + Images[j], 0)
            img = cv2.resize(img, (64, 64))
            TrainingdataFrame = TrainingdataFrame.append({'Image': img, 'Label': i}, ignore_index=True)

    ClassPath = TestPath
    Images = os.listdir(ClassPath)
    #print("images in class ", NumberOfClasses[i], " are ", Images)
    for j in range(len(Images)):
        # print("image name ", Images[j])
        # print("image path ", TestPath + Images[j])
        img = cv2.imread(TestPath + Images[j], 0)
        img = cv2.resize(img, (64, 64))
        TestdataFrame = TestdataFrame.append({'Image': img}, ignore_index=True)

    return TrainingdataFrame, TestdataFrame

# Main function
if __name__ == '__main__':
    path = '/Users/anushsriramramesh/Downloads/archive'
    TrainingdataFrame, TestdataFrame = load_data(path)

    trainingDataImages = np.array(TrainingdataFrame['Image'])
    print("trainingDataImages shape ", trainingDataImages.shape)
    print(TrainingdataFrame)
    print(TestdataFrame)

    TrainingdataFrame.to_csv('TrainingdataFrame.csv')
    TestdataFrame.to_csv('TestdataFrame.csv')