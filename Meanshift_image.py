import cv2 as cv
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import pandas as pd


class meanshift:
    def __init__(self):
        pass

    def image_rgb(self, path):
        '''
        Reads and returns RGB image from the path
        '''
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img

    def image_label(self, img):
        '''
        Turn RGB label image into grayscale with 1 and 9
        '''
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img = np.where(img == 0, 0, 1)
        return img


    def meanshifted_img(self, img, debug = 0):
        '''
        Return the results of mean shift algorithm on an image
        '''
        imgX = img.astype('float')
        y, x, _ = img.shape
        ycoord, xcoord = np.meshgrid(range(y), range(x), indexing='ij')
        ycoord = ycoord.reshape((y, x, 1))
        xcoord = xcoord.reshape((y, x, 1))

        imgX = np.concatenate((imgX, ycoord, xcoord), axis = 2)

        flat_img = np.reshape(imgX, [-1, 5])

        bandwidth = estimate_bandwidth(flat_img, quantile=0.05, n_samples=300)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding = True, n_jobs=-1)

        ms.fit(flat_img)

        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        segmented_img = cluster_centers[np.reshape(labels, img.shape[:2])][:,:,:3]
        if debug == 1:
            return ms, segmented_img.astype(np.uint8)
        else:
            return ms

    
