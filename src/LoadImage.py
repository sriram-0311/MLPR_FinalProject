import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import datasets
import pandas as pd
import os
import warnings
import tensorflow as tf

# load data from given path using tf keras dataloader
def load_data_tf(path):
    # load data from given path
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="grayscale",
        image_size=(64, 64),
        shuffle=True,
        batch_size=8700,
        seed=123,
        validation_split=0.2,
        subset="validation",
        interpolation="bilinear",
        follow_links=False,
    )
    return train_data


# Suppress warnings
warnings.filterwarnings("ignore")

#print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))

# load image sample data using sklearn datasets module
# def load_data_sklearn():


# Load training and test data from given path
def load_data(path):
    TrainingdataFrame = pd.DataFrame(columns=['Image', 'Label'])
    TestdataFrame = pd.DataFrame(columns=['Image'])
    TrainingImages = np.empty((87000, 64,64), dtype=np.uint8)
    TrainingLabels = np.empty((0, 1), dtype=np.uint8)

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
            # cv2.imshow('image', img)
            # cv2.waitKey(0)
            print("img shape ", img.shape, " img type ", type(img), "\t",i, "\t", j)
            # TrainingImages = np.vstack((TrainingImages, [img]))
            # TrainingLabels = np.append(TrainingLabels, i)
            TrainingdataFrame = TrainingdataFrame.append({'Image': img, 'Label': i}, ignore_index=True)

    ClassPath = TestPath
    Images = os.listdir(ClassPath)
    #print("images in class ", NumberOfClasses[i], " are ", Images)
    for j in range(len(Images)):
        # print("image name ", Images[j])
        # print("image path ", TestPath + Images[j])
        img = cv2.imread(TestPath + Images[j], 0)
        img = cv2.resize(img, (64, 64))
        #TestdataFrame = TestdataFrame.append({'Image': img}, ignore_index=True)

    return TrainingImages, TrainingLabels

# return 

# Main function
if __name__ == '__main__':
    path = '/Users/anushsriramramesh/Downloads/archive/asl_alphabet_train/asl_alphabet_train'
    #TrainingdataFrame, TestdataFrame = load_data(path)
    #TrainingdataImages, TestdataLabel = load_data(path)

    # np.savetxt("TrainingdataImages.csv", TrainingdataImages)
    # np.savetxt("TestdataLabel.csv", TestdataLabel)

    #trainingDataImages = np.array(TrainingdataFrame['Image'])
    # show a random image from training data
    # plt.imshow(TrainingdataImages[0])
    # plt.imshow(TrainingdataImages[np.random.randint(0,len(TrainingdataImages))], cmap='gray')
    # plt.show()
    # print("trainingDataImages shape ", trainingDataImages.shape)
    # print(TrainingdataFrame)
    # print(TestdataFrame)

    # TrainingdataFrame.to_csv('TrainingdataFrame.csv')
    # TestdataFrame.to_csv('TestdataFrame.csv')

    # read data using tf keras dataloader
    train_data = load_data_tf(path)
    print(train_data)
    print(train_data.class_names)
    image_batch, label_batch = next(iter(train_data))
    trainData = image_batch.numpy()
    trainLabel = label_batch.numpy()
    print("trainData shape ", trainData.shape)
    print("trainLabel shape ", trainLabel.shape)
    print(image_batch.shape)
    print(label_batch.shape)

    # display one image from training data
    # image = train_data[0]
    # plt.imshow(image[0])
    # plt.show()