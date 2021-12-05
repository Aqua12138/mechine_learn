from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import  KNeighborsClassifier
#1 获取数据
iris = load_iris()

#2 数据基本处理
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=2)
#3 特征工程  -  特征预处理
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)
#4 机器学习
#4.1 实例化估计器
est = KNeighborsClassifier(n_neighbors=5,algorithm='kd_tree')
#4.2 模型调优 -- 交叉验证
param_grid = {'n_neighbors':[1,3,5,7]}
est = GridSearchCV(est,param_grid=param_grid,cv=5)
#4.3 模型训练
est.fit(x_train,y_train)
#5 模型评估
#5.1 预测值输出
y_pre = est.predict(x_test)
print('预测值是：\n',y_pre)
print('预测值和真是值的对比是：\n',y_pre==y_test)
#5.2 准确率计算
score = est.score(x_test,y_test)
print(score)
print(y_train)
#5.3 查看交叉验证 网格搜索属性
print('在交叉验证中得到的最好结果是：\n',est.best_score_)
print('在交叉验证中得到的最好模型是：\n',est.best_estimator_)
print('在交叉验证中得到的模型结果是：\n',est.cv_results_)#准确率可靠性稳定性
