import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs#用于生成聚集的随机点
from sklearn.cluster import KMeans#聚类算法
from sklearn.metrics import calinski_harabasz_score#评估聚类算法的好坏 越大越好
#创建数据
x,y = make_blobs(n_samples=1000,n_features=2,centers=[[-1,1],[0,0],[1,1],[2,2]],cluster_std=[0.4,0.2,0.2,0.2],random_state=0)#x用来记录数据的特征值，y记录每一个数据属于的簇
# plt.scatter(x[:,0],x[:,1],marker='o')
#kmeans训练，可视化
pre = KMeans(n_clusters=4,random_state=9).fit_predict(x)#n_clusters=表示分类的组数
#可视化展示
plt.scatter(x[:,0],x[:,1],c=pre)#c是按照点分类
print(calinski_harabasz_score(x,pre))#评估
plt.show()
