import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier


#1.获取数据
taitan = pd.read_csv('/Users/app/Desktop/titanic/train.csv')
#2.数据基本处理
#2.1pclass one-hot编码
zeros = np.zeros([taitan.shape[0],3])
pclass = pd.DataFrame(zeros,columns=['pclass=1','pclass=2','pclass=3'])
for i in range(taitan.shape[0]):
    pclass.loc[i,'pclass={}'.format(taitan['Pclass'][i])]=1
new_taitan = pd.concat([taitan,pclass],axis=1)
#2.2 确定特征值，目标值
x = new_taitan[['pclass=1','pclass=2','pclass=3','Age','Sex']]
y = new_taitan['Survived']#单括号Series 双括号DataFrame
#2.3缺失值处理
x['Age'].fillna(taitan['Age'].mean(),inplace=True)
#2.4数据集分割
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=22,test_size=0.2)
#3.特征工程(字典特征提取）
x_train = x_train.to_dict(orient='records')#转换成字典格式
x_test = x_test.to_dict(orient = 'records')
transfer = DictVectorizer(sparse=False)
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)
names = transfer.get_feature_names()

#4.机器学习（随机森林+cv）
estimator = RandomForestClassifier()
param_grid = {'n_estimators':[120,200,300,500,800],'max_depth':[5,8,10,15,18]}
estimator = GridSearchCV(estimator,param_grid=param_grid,cv=3)
estimator.fit(x_train,y_train)
#5.模型评估
y_pre = estimator.predict(np.array([[10,1,0,1,0,0]]))
ret = estimator.score(x_test,y_test)
best = estimator.best_estimator_
print(ret)
print(y_pre)
print('最好的模型',best)
#export_graphviz(estimator,out_file='/Users/app/Desktop/tree.dot',feature_names=['Age', 'Sex=female', 'Sex=male', 'pclass=1', 'pclass=2', 'pclass=3'])
