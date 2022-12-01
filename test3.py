import pickle
import numpy as np

clf = pickle.load(open('ensemble_random_forest_model.sav', 'rb'))

from ensemble_feature import ensemble_method_feature
emf = ensemble_method_feature()
img = emf.image_resize(emf.image_rgb('Hieu Thesis dataset ABB modified/SL2_sa-ptz_1622547888709965.png'), scale=0, height=120, width = 120)
f = emf.combine_feature(img)
print(f.shape)
f = np.reshape(f,(1, f.size))
print(f.shape)
clf.predict(f)