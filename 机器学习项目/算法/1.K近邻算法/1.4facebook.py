import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import  KNeighborsClassifier

data = pd.read_csv('/Users/app/Downloads/day05-资料/2.code/data/FBlocation/train.csv')
print(data.columns)
#2.1 筛选数据
data_q = data.query('x>2.0 & x<2.5 & y>2.0 & y<2.5')
#2.2 将时间转化为标准格式
time = pd.to_datetime(data_q['time'],unit='s')#声明时间单位是秒
time = pd.DatetimeIndex(time)#将time从DateFrame格式 转化成 DatetimeIndex，可以提取处相应的月份 年份 小时等
data_q['hour'] = time.hour
data_q['day'] = time.day
data_q['weekday'] = time.weekday
#2.3 去掉签到少的地方
place_count = data_q.groupby('place_id').count()
place_count = place_count[place_count['row_id']>3]#提取大于3次的数据的count列表
print(place_count)
data_q = data_q[data_q['place_id'].isin(place_count.index)]#比较提取前和提取后index，并删除没有的
#2.4 提取特征值和目标值
x = data_q[['x','y','accuracy','hour','day','weekday']]#特征值
y = data_q['place_id']#目标值
#2.5 分割数据集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=22)
#3 特征工程  -  特征预处理
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)
#4 机器学习
#4.1 实例化估计器
est = KNeighborsClassifier(n_neighbors=5,algorithm='kd_tree')
#4.2 模型调优 -- 交叉验证
param_grid = {'n_neighbors':[3,5,7,9]}
est = GridSearchCV(estimator=est,param_grid=param_grid,cv=3,n_jobs=4)#n_jobs 需要跑几个cpu -1表示全部cpu 一共8个
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



