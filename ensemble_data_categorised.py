import os
from file_reader import FileReader
from sklearn.model_selection import train_test_split
import numpy as np
from ensemble_feature import ensemble_method_feature

path = 'dataset2'
ensemble = ensemble_method_feature()

images = FileReader.files(path)

cloudy_path = 'dataset2/cloudy'
rain_path = 'dataset2/rain'
shine_path = 'dataset2/shine'
sunrise_path = 'dataset2/sunrise'

if not os.path.exists(cloudy_path):
    os.makedirs(cloudy_path)

if not os.path.exists(rain_path):
    os.makedirs(rain_path)

if not os.path.exists(shine_path):
    os.makedirs(shine_path)

if not os.path.exists(sunrise_path):
    os.makedirs(sunrise_path)

for image in images:
    name = image.split('/')[-1]
    if 'cloudy' in name:
        os.rename(image, cloudy_path + '/' + name)
    elif 'rain' in name:
        os.rename(image, rain_path + '/' + name)
    elif 'shine' in name:
        os.rename(image, shine_path + '/' + name)
    elif 'sunrise' in name:
        os.rename(image, sunrise_path + '/' + name)


cloudy_imgs = FileReader.files(cloudy_path)
rain_imgs = FileReader.files(rain_path)
shine_imgs = FileReader.files(shine_path)
sunrise_imgs = FileReader.files(sunrise_path)

cloudy_data = [(name, 0) for name in cloudy_imgs]
rain_data = [(name, 1) for name in rain_imgs]
shine_data = [(name, 2) for name in shine_imgs]
sunrise_data = [(name, 3) for name in sunrise_imgs]

X_cloud = [x[0] for x in cloudy_data]
y_cloud = [x[1] for x in cloudy_data]


X_rain = [x[0] for x in rain_data]
y_rain = [x[1] for x in rain_data]


X_shine = [x[0] for x in shine_data]
y_shine = [x[1] for x in shine_data]


X_sunrise = [x[0] for x in sunrise_data]
y_sunrise = [x[1] for x in sunrise_data]


# Create training and testing data

Xtr_cloud, Xtst_cloud, ytr_cloud, ytst_cloud = train_test_split(X_cloud, y_cloud, test_size = 0.33, random_state = 42)
Xtr_rain, Xtst_rain, ytr_rain, ytst_rain = train_test_split(X_rain, y_rain, test_size = 0.33, random_state = 42)
Xtr_shine, Xtst_shine, ytr_shine, ytst_shine = train_test_split(X_shine, y_shine, test_size = 0.33, random_state = 42)
Xtr_sunrise, Xtst_sunrise, ytr_sunrise, ytst_sunrise = train_test_split(X_sunrise, y_sunrise, test_size = 0.33, random_state = 42)

Xtr = Xtr_cloud + Xtr_rain + Xtr_shine + Xtr_sunrise
Xtst = Xtst_cloud + Xtst_rain + Xtst_shine + Xtst_sunrise
ytr = ytr_cloud + ytr_rain + ytr_shine + ytr_sunrise
ytst = ytst_cloud + ytst_rain + ytst_shine + ytst_sunrise

# Shuffling data for randomness
tr = list(zip(Xtr, ytr))
np.random.shuffle(tr)

Xtr = [x[0] for x in tr]
ytr = np.array([x[1] for x in tr])

tst = list(zip(Xtst, ytst))
np.random.shuffle(tst)

Xtst = ([x[0] for x in tst])
ytst = np.array([x[1] for x in tst])

Xtr_img = [ensemble.image_resize(ensemble.image_rgb(x), scale=0, height=120, width = 120) for x in Xtr]
Xtr_vector = np.array([ensemble.combine_feature(img) for img in Xtr_img])

Xtst_img = [ensemble.image_resize(ensemble.image_rgb(x), scale=0, height=120, width = 120) for x in Xtst]
Xtst_vector = np.array([ensemble.combine_feature(img) for img in Xtst_img])

with open('Xtr_vector.npy', 'wb') as f:
    np.save(f,Xtr_vector)

with open('Xtst_vector.npy', 'wb') as f:
    np.save(f,Xtst_vector)

with open('ytr.npy', 'wb') as f:
    np.save(f,ytr)

with open('ytst.npy', 'wb') as f:
    np.save(f,ytst)
