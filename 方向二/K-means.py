#加载K-Means库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs



#生成测试样本
n_samples=1500
random_state=170

#用来生成聚类算法的测试数据
# make_blobs会根据用户指定的特征数量、中心点数量、范围等来生成几类数据，这些数据可用于测试聚类算法的效果。
#n_samples:待生成的样本的总数
#random_state:随机生成器的种子
#n_features是每个样本的特征数。
#centers表示类别数。
#cluster_std表示每个类别的方差，例如我们希望生成2类数据，其中一类比另一类具有更大的方差，可以将cluster_std设置为[1.0,3.0]。

x,y=make_blobs(n_samples=n_samples,random_state=random_state)
#进行聚类指定聚类个数
y_pred=KMeans(n_clusters=4,random_state=random_state).fit_predict(x)

#可视化
# plt.subploit(221) # 第一行的左图
plt.scatter(x[:,0],x[:,1],c=y_pred)
plt.title("K-means test")
plt.show()