import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
#1 获取数据
max_min = pd.read_csv('/Users/app/Desktop/max-min.csv',usecols=['max','min'],encoding='unicode_escape')
fs = pd.read_csv('/Users/app/Desktop/辐射.csv',usecols=['fs'],encoding='unicode_escape')
target = pd.read_csv('/Users/app/Desktop/降水量.csv',usecols=['rainnum'],encoding='unicode_escape')
print(type(target))
data = pd.concat([max_min,fs] , axis = 1)
target = target.values#记得数据运算要从DateFrame转化成array
target = np.where(target > 0,1,0)
#2 数据预处理
x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=22)
#3 特征预处理
est = StandardScaler()
x_train = est.fit_transform(x_train)
x_test = est.fit_transform(x_test)
print(type(y_test))
#4 机器学习
gj = KNeighborsClassifier(n_neighbors=5,algorithm='kd_tree')
gj.fit(x_train,y_train)
#5 模型评估)
y_pre = gj.predict(x_test)
print('预测值是：\n',y_pre)
print('预测值和真实值的对比是：\n',y_pre==y_test)
score = gj.score(x_test,y_test)
print(score)