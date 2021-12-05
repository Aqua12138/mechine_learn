import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
# 获取数据
order_product = pd.read_csv('/Users/app/Downloads/day07-资料/2.code/data/instacart/order_products__prior.csv')
product = pd.read_csv('/Users/app/Downloads/day07-资料/2.code/data/instacart/products.csv')
order = pd.read_csv('/Users/app/Downloads/day07-资料/2.code/data/instacart/orders.csv')
aisles = pd.read_csv('/Users/app/Downloads/day07-资料/2.code/data/instacart/aisles.csv')
# 数据合并
table1 = pd.merge(order_product,product,on=['product_id','product_id'])
table2 = pd.merge(table1,order,on=['order_id','order_id'])
table = pd.merge(table2,aisles,on=['aisle_id','aisle_id'])

# 交叉表合并
data = pd.crosstab(table['user_id'],table['aisle'])
# 数据截取
new_data = data.iloc[:1000,:]
transfer = PCA(n_components=0.9)#特征降维，减少数据
trans_data = transfer.fit_transform(new_data)
#机器学习
estimator = KMeans(n_clusters=5)
y_pre = estimator.fit_predict(trans_data)
#模型评估
ret = silhouette_score(trans_data,y_pre)
ret2 = calinski_harabasz_score(trans_data,y_pre)
print(ret,ret2)
