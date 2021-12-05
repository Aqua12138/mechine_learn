import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
def minmax_demo():
    #归一化 对于异常点的抵抗性低（鲁棒性差）
    data = pd.read_csv('/Users/app/Downloads/day04-资料/2.code/data/dating.txt')
    print(data)
    #1.实例化
    transform = MinMaxScaler(feature_range=(3,5))
    #2.进行转换，调用fit_transform
    ret_data = transform.fit_transform(data[['milage','Liters','Consumtime']])
    print('归一化之后的数据：\n',ret_data)
# minmax_demo()
def stand_demo():
    #标准化 鲁棒性高
    data = pd.read_csv('/Users/app/Downloads/day04-资料/2.code/data/dating.txt')
    print(data)
    #1.标准化
    transform = StandardScaler()
    #2.进行转换，调用fit_transform
    ret_data = transform.fit_transform(data[['milage','Liters','Consumtime']])
    print('标准化之后的数据：\n',ret_data)
    print('每一列的方差为：\n',transform.var_)
    print('每一列的平均值为：\n',transform.mean_)
stand_demo()