import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('two_class_df.csv')

feature_cols = ['max t', 'std t', 'mean t', 'mean red', 'mean green', 'mean blue', 'combine mean rgb', 'median of mean rgb', 'mean intensity', 'mean gradient', 'lightness', 'meanX', 'meanY']
X = df.loc[:, feature_cols]
y = df['y']

Xtr, Xtst, ytr, ytst = train_test_split(X, y, test_size = 0.33)

rfclf = RandomForestClassifier(random_state=0)
rfclf.fit(Xtr, ytr)
rfacc = rfclf.score(Xtst, ytst)
print('Accuracy random forest:',rfacc)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xtr, ytr)
knnacc = knn.score(Xtst, ytst)
print('Accuracy Knn:', knnacc)

svc = SVC(kernel = 'rbf', degree = 1, random_state = 0)
svc.fit(Xtr, ytr)
svcacc = svc.score(Xtst, ytst)
print('Accuracy SVC:', svcacc)

# pickle.dump(clf, open('sky_random_forest_model.sav', 'wb'))

