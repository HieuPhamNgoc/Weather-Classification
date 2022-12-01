import numpy as np
from sky_segmentation_feature import sky_segmentation
from Meanshift_image import meanshift
from file_reader import FileReader
from datetime import datetime
import pickle

start_time = datetime.now()
ms = meanshift()
ss = sky_segmentation()
clf = pickle.load(open('sky_random_forest_model.sav', 'rb'))
images = FileReader.files('Hieu Thesis dataset ABB modified')

images_sky_area_X = []
images_sky_area_Y = []

for img_path in images:
    img = ms.image_rgb(img_path)
    img_sky_areas_X = []
    img_sky_areas_Y = []
    feature, posY, posX = ss.combine_feature_vector(img)
    ypred = clf.predict(feature)
    skysegid = np.where(ypred == 1)
    for id in skysegid[0]:
        if (0 in posY[id]):
            img_sky_areas_X.append(posX[id])
            img_sky_areas_Y.append(posY[id])
    images_sky_area_X.append(img_sky_areas_X)
    images_sky_area_Y.append(img_sky_areas_Y)

pickle.dump(images_sky_area_X, open('images_sky_area_X.data', 'wb'))
pickle.dump(images_sky_area_Y, open('images_sky_area_Y.data', 'wb'))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))