from Meanshift_image import meanshift
import matplotlib.pyplot as plt
import numpy as np
from sky_segmentation_feature import sky_segmentation
from datetime import datetime
import pickle
import pandas as pd

clf = pickle.load(open('sky_random_forest_model.sav', 'rb'))
ss = sky_segmentation()

def main():
    path = input('Enter the image path:\n')
    img = ss.image_rgb(path)

    feature, posY, posX = ss.combine_feature_vector(img)


    ypred = clf.predict(feature)

    skysegid = np.where(ypred == 1)
    plt.imshow(img)
    for id in skysegid[0]:
        if (0 in posY[id]):
            plt.scatter(posX[id], posY[id], s=5,c='white')
        else:
            continue
    plt.show()
main()
