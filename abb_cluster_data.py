from Meanshift_image import meanshift
import matplotlib.pyplot as plt
import numpy as np
from sky_segmentation_feature import sky_segmentation
from datetime import datetime
import pickle
import pandas as pd
from sky_region_feature import sky_region
from file_reader import FileReader

ms = meanshift()

images_sky_area_X = pickle.load(open('images_sky_area_X.data','rb'))
images_sky_area_Y = pickle.load(open('images_sky_area_Y.data','rb'))
images = FileReader.files('Hieu Thesis dataset ABB modified')

n_images = len(images)

cluster_features = []

for img_i in range(n_images):
    
    img = ms.image_rgb(images[img_i])
    regionX = images_sky_area_X[img_i]
    regionY = images_sky_area_Y[img_i]    

    sr = sky_region(img, regionX, regionY)

    colorfulness = sr.colorfulness()
    intensity = sr.intensity()
    avg_gradient = sr.avg_gradient()
    CEL, CEa, CEb = sr.contrast_energy()

    img_features = [colorfulness, intensity, avg_gradient, CEL, CEa, CEb]

    cluster_features.append(img_features)

pickle.dump(cluster_features, open('abb_cluster_features(with ce).data', 'wb'))