#preprocessing functions for our image data
import cv2

def regularization()

def PCA()

def threshold_image(image):
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

    # global thresholding
    ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    # Otsu's thresholding
    ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    return th3

def blob_detection(im):
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector()
    
    # Detect blobs.
    keypoints = detector.detect(im)
    return keypoints
def plot_image():
    # cv2.imshow('Black white image', blackAndWhiteImage)
    # cv2.imshow('Original image',originalImage)
    # cv2.imshow('Gray image', grayImage)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()