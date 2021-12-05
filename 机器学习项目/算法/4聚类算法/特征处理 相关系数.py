import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import  pearsonr,spearmanr
from sklearn.decomposition import PCA#特征降维 高维变低维，转化为新的维度
def var_thr():
    #特征选择低方差过滤
    data = pd.read_csv('/Users/app/Downloads/day07-资料/2.code/data/factor_returns.csv')
    print(data.shape,data.columns)
    transfer = VarianceThreshold(threshold=1)#方差小于1都不要
    #转换
    new_data = transfer.fit_transform(data.iloc[:,1:10])
    print(data)
    print(pd.DataFrame(new_data))
def pea_demo():
    #皮尔逊
    x1 = np.array([12.4,15.3,23,26,33,34,39,45,55,60])
    x2 = [21,23,32,34,42,43,49,52,59,63]
    ret = pearsonr(x1,x2)
    print('皮尔逊相关系数的结果是：\n',ret)
def spea_demo():
    #皮尔逊
    x1 = [12.4,15.3,23,26,33,34,39,45,55,60]
    x2 = [21,23,32,34,42,43,49,52,59,63]
    ret = spearmanr(x1,x2)
    print('斯皮尔曼相关系数的结果是：\n',ret)

def pca_demo():
    data = [[2,8,4,5],[6,3,0,8],[5,4,9,1]]
    transfer = PCA(n_components=0.9)#小数表示为保留百分之多少，整数是保留的维度
    trans_data = transfer.fit_transform(data)
    print('保留0。9的数据后的维度是：\n',trans_data)

# var_thr()
pea_demo()#(0.9936811071785147, 6.922214954626871e-09)第一个值越接近1越相关，第二个值越接近0，越相关
# spea_demo()
# pca_demo()