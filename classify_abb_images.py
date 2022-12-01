import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = pickle.load(open('abb_cluster_features.data', 'rb'))
X = np.array([np.array(x) for x in X])

y = pd.read_excel('ABB modified dataset weather label.xlsx')
y = np.array(y).ravel()

i1 = np.where(y == 1)
X1 = X[i1]
y1 = y[i1]
X1tr, X1tst, y1tr, y1tst = train_test_split(X1, y1, test_size = 0.33)


i2 = np.where(y == 2)
X2 = X[i2]
y2 = y[i2]
X2tr, X2tst, y2tr, y2tst = train_test_split(X2, y2, test_size = 0.33)

i3 = np.where(y == 3)
X3 = X[i3]
y3 = y[i3]
X3tr, X3tst, y3tr, y3tst = train_test_split(X3, y3, test_size = 0.33)

Xtr = np.concatenate([X1tr, X2tr, X3tr])
ytr = np.concatenate([y1tr, y2tr, y3tr])
Xtst = np.concatenate([X1tst, X2tst, X3tst])
ytst = np.concatenate([y1tst, y2tst, y3tst])

random_forest_clf = RandomForestClassifier(random_state = 0)
random_forest_clf.fit(Xtr, ytr)
print('Random forest score: ', random_forest_clf.score(Xtst, ytst))

svc = SVC(kernel = 'linear', probability = True)
svc.fit(Xtr, ytr)
print('Linear SVM score:', svc.score(Xtst, ytst))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xtr, ytr)
print('K-NN score:', knn.score(Xtst, ytst))