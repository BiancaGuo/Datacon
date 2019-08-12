#DBSCAN(eps=0.5,min_samples=5,metric='euclidean',algorithm='auto',leaf_size=30/p=None,n_jobs=1)
'''
eps:同一个聚类集合中两个样本的最大距离
min_samples:同一个聚类集合中最小样本数
algorithm："auto"/"ball_tree"/"kd_tree"/"brute"
leaf_size：使用BallTree或cKDTree算法时叶子结点个数
n_jobs：并发任务数
'''
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn import metrics
#特征缩放
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def show_dbscan():
    centers=[[1,1],[-1,-1],[1,-1]]
    X,labels_true=make_blobs(n_samples=10000,cluster_std=0.4,random_state=0)
    X=StandardScaler().fit_transform(X)
    db=DBSCAN(eps=0.5,min_samples=200)
    #按照某一个已知的数组的规模（几行几列）建立同样规模的特殊数组
    y = db.fit_predict(X)
    # print(db.labels_)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

if __name__ == '__main__':
    show_dbscan()