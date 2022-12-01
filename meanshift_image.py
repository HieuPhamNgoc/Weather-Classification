import cv2 as cv
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from Image_segmentation import meanshift
import matplotlib.pyplot as plt
def main():
    path = input('Input image path:\n')
    m = meanshift()

    img = m.image_rgb(path).astype('float')
    ms, seg_img = m.meanshifted_img(img, debug = 1)

    plt.imshow(seg_img)
    plt.show()

main()

# abb\EF_sa-camera-1_1624708062360070.png
