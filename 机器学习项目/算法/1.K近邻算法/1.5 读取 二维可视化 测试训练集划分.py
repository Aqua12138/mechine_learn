from sklearn.datasets import load_iris,fetch_20newsgroups
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split#划分数据集和测试集
# 1.获取数据
# 1.1 获取本地数据
iris = load_iris()
print(iris)
# 1.2 获取大数据集
new = fetch_20newsgroups()
print(new)
# 2.数据集属性描述
print('数据集特征值是：{}\n'.format(iris.data))
print('数据集目标值是：{}\n'.format(iris['target']))
print('数据集的特征名字是：{}\n'.format(iris.feature_names))
print('数据集的目标值名字是：{}\n'.format(iris.target_names))
print('数据集的描述：{}\n'.format(iris.DESCR))
# 3.数据可视化
iris_d =  pd.DataFrame(data = iris.data,columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'])
iris_d['target'] = iris.target
print(iris_d)
def iris_plot(data, col1 ,col2):
    sns.lmplot(x=col1 ,y=col2 ,data = data,hue ='target',fit_reg = False)
    plt.show()
iris_plot(iris_d,'Sepal_Width','Sepal_Length')
iris_plot(iris_d,'Petal_Width','Petal_Length')
#4.据集的划分
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
print('训练集的特征值是：\n',x_train)
print('训练集的目标值是：\n',y_train)
print('测试集的特征值是：\n',x_test)
print('测试集的目标值是：\n',y_test)
print('训练集的特征值形状是：\n',x_train.shape)
print('训练集的目标值形状是：\n',y_train.shape)
print('测试集的特征值形状是：\n',x_test.shape)
print('测试集的目标值形状是：\n',y_test.shape)