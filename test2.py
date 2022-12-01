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

img = ms.image_rgb(images[1])
regionX = images_sky_area_X[1]
regionY = images_sky_area_Y[1]

sr = sky_region(img, regionX, regionY)

colorfulness = sr.colorfulness()
intensity = sr.intensity()
avg_gradient = sr.avg_gradient()

