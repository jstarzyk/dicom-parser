import cv2
import numpy as np

def adaptiveThreshold(image, size=31, c=4):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, size, c)

neg = np.vectorize(lambda x: 0 if x == 255 else 1)

def morphologyEx(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    bin_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel)
    
def distanceTransform(image):
    dist = cv2.distanceTransform(image, cv2.DIST_L2, 5)
    return dist, cv2.normalize(np.copy(dist), np.copy(dist), 0, 255, cv2.NORM_MINMAX)
    
def laplacian(image):
    lapl = cv2.Laplacian(np.uint8(image), cv2.CV_64FC1, ksize=7)
    return cv2.normalize(lapl, lapl, 0, 255, cv2.NORM_MINMAX)
    
def threshold(image, thr=90):
    return cv2.threshold(np.uint8(image), thr, 255, cv2.THRESH_BINARY_INV)[1]
    
def get_bin_image(image):
    th = adaptiveThreshold(image)
    bin_image = morphologyEx(np.uint8(neg(th)))
    dist, norm_dist = distanceTransform(bin_image)
    lapl = laplacian(norm_dist)
    return dist, neg(threshold(lapl))
