#preprocessing functions for our image data
import cv2
import pandas as pd
import numpy as np

# Read image from csv file
def read_image_from_csv(csv_file):
    # Read csv file
    df = pd.read_csv(csv_file)
    # get each image from the csv file
    images = df['Image']
    print("images :", images[0][0])
    arrayformed = np.fromstring(images[0], dtype=int, sep=' ').reshape(64,64)
    print("array formed ",arrayformed)
    # convert each image to numpy array
    images = [np.fromstring(image, dtype=int, sep='\n') for image in images]
    print("images shape ", np.array(images).shape)
    print("images ", images[0])
    # Convert string to numpy array
    df['Image'] = df['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))
    # Reshape image
    #df['Image'] = df['Image'].apply(lambda x: x.reshape(64,64))
    return df

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
    df = read_image_from_csv('TrainingdataFrame.csv')
    # Show a random image
    img = df['Image'][np.random.randint(0,len(df['Image']))]
    # img = threshold_image(img)
    keypoints = blob_detection(img)
    print(keypoints)
    plot_image(img)