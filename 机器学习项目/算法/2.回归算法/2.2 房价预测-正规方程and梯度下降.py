from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor,RidgeCV,Ridge
from sklearn.metrics import mean_squared_error
def linear_model1():
    #1.数据获取
    boston = load_boston()
    #2.数据基本处理
    #2.1 分割数据
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,test_size=0.2)
    #3.特征工程--标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    #4.机器学习-线性回归
    estimator = LinearRegression()
    estimator.fit(x_train,y_train)

    print('这个模型的偏置是：\n',estimator.intercept_)
    print('这个模型的系数是：\n',estimator.coef_)
    #5.模型评估
    y_pre = estimator.predict(x_test)
    print('预测值：\n',y_pre)

    ret = mean_squared_error(y_test,y_pre)
    print('均方误差：\n',ret)




def linear_model2():
    #1.数据获取
    boston = load_boston()
    #2.数据基本处理
    #2.1 分割数据
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,test_size=0.2)
    #3.特征工程--标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    #4.机器学习-梯度下降
    estimator = SGDRegressor(max_iter=1000,learning_rate='constant',eta0=0.007)
    estimator = SGDRegressor(max_iter=1000)
    estimator.fit(x_train,y_train)

    print('这个模型的偏置是：\n',estimator.intercept_)
    print('这个模型的系数是：\n',estimator.coef_)
    #5.模型评估
    y_pre = estimator.predict(x_test)
    print('预测值：\n',y_pre)

    ret = mean_squared_error(y_test,y_pre)
    print('均方误差：\n',ret)
def linear_model3():
    # 1.数据获取
    boston = load_boston()
    # 2.数据基本处理
    # 2.1 分割数据
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)
    # 3.特征工程--标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    # 4.机器学习-梯度下降
    #estimator = Ridge(alpha=1.0)
    estimator = RidgeCV(alphas = (0.001,0.01,0.1,10,100))
    estimator.fit(x_train, y_train)

    print('这个模型的偏置是：\n', estimator.intercept_)
    print('这个模型的系数是：\n', estimator.coef_)
    # 5.模型评估
    y_pre = estimator.predict(x_test)
    print('预测值：\n', y_pre)

    ret = mean_squared_error(y_test, y_pre)
    print('均方误差：\n', ret)
#linear_model1()#线性回归
#linear_model2()#随机梯度下降
linear_model3()#随机平均梯度下降+损失函数岭回归