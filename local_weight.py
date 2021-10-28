import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import LeaveOneOut
import numpy as np

df = pd.read_csv("glcm_texture_feature.csv")
X = df.iloc[:,:-2]
Y = df.iloc[:,-1]

kmeans = KMeans(n_clusters= 8, random_state=0).fit(X)

cluster_map = pd.DataFrame()
cluster_map['image'] = Y
cluster_map['cluster'] = kmeans.labels_
for i in range(0,8):
    print(cluster_map[cluster_map['cluster']==i])




df['cluster'] = kmeans.labels_
cv = LeaveOneOut()


for i in range(8):
    train = df[df['cluster']==i]
    
    
    for train_ix, test_ix in cv.split(train):

        train = df.iloc[train_ix, :]
        test = df.iloc[test_ix,:]

        instance_name = list(test['image'])[0]
        instance_feature = test.to_numpy()[:,:-2]


        data_point_names = list(train['image'])
        data_point_feature = train.to_numpy()[:,:-2]

        kernel_width = 0.25
        distances = sklearn.metrics.pairwise_distances(instance_feature, data_point_feature, metric='cosine')
        weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function
