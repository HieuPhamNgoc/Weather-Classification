from Meanshift_image import meanshift
import matplotlib.pyplot as plt
import numpy as np
from file_reader import FileReader
from sky_segmentation_feature import sky_segmentation
import pandas as pd
from datetime import datetime

start_time = datetime.now()
ms = meanshift()
ss = sky_segmentation()

labels = FileReader.files('skyfinder_dataset/Label')

data_folders = FileReader.files('skyfinder_dataset/Data')

df = pd.DataFrame(columns=['max t', 'std t', 'mean t', 'mean red', 'mean green', 'mean blue', 'combine mean rgb', 'median of mean rgb', 'mean intensity', 'mean gradient', 'lightness','meanX', 'meanY','y'])

for data_folder, label in list(zip(data_folders, labels)):
    imgY = ms.image_label(ms.image_rgb(label))
    image_paths = FileReader.files(data_folder)
    image_paths = image_paths[0:10]
    for path in image_paths:
        img = ms.image_rgb(path)
        dff = ss.combine_feature_dataframe(img, imgY)
        df = df.append(dff)

df.to_csv('two_class_df.csv')
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))



