import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ensemble_feature import ensemble_method_feature
emf = ensemble_method_feature()

Xtr = np.load('Xtr_vector_ensemble.npy')
ytr = np.load('ytr_ensemble.npy')
Xtst = np.load('Xtst_vector_ensemble.npy')
ytst = np.load('ytst_ensemble.npy')

clf = RandomForestClassifier(random_state=0)
clf.fit(Xtr, ytr)
print(clf.score(Xtst, ytst))

img = emf.image_resize(emf.image_rgb('Hieu Thesis dataset ABB modified/SL2_sa-ptz_1622547888709965.png'), scale=0, height=120, width = 120)
f = emf.combine_feature(img)
print(f.shape)
f = np.reshape(f,(1, f.size))
print(f.shape)
print(clf.predict(f))
