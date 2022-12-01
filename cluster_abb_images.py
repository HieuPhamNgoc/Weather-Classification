import pickle
from sklearn.cluster import KMeans
import numpy as np
from file_reader import FileReader

cluster_features = pickle.load(open('abb_cluster_features.data', 'rb'))
cluster_features = np.array([np.array(x) for x in cluster_features])

cluster_features_ce = pickle.load(open('abb_cluster_features(with ce).data', 'rb'))
cluster_features_ce = np.array([np.array(x) for x in cluster_features_ce])

kmeans = KMeans(n_clusters = 3, random_state = 0).fit(cluster_features)

print(kmeans.labels_)

images = FileReader.files('Hieu Thesis dataset ABB modified')
imgs_name = [x.split('/')[-1] for x in images]

clusters = dict(zip(imgs_name, kmeans.labels_))

print(clusters['SL2_sa-ptz_1619335611572997.png'])

