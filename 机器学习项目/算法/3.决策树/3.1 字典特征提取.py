from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import jieba

def dict_demo():
    #字典特征提取
    #1.获取数据
    data = [{'city':'北京','temperature':100},{'city':'上海','temperature':60},{'city':'深圳','temperature':30}]
    #2. 字典特征提取
    #2.1 实例化
    transfer = DictVectorizer(sparse=False)#是否变成sparse矩阵sparse=True 节省空间
    #2.2 转换
    new_data = transfer.fit_transform(data)
    print(new_data)
    names = transfer.get_feature_names()
    print('属性名字是：\n',names)


def englis_demo():
    #英文文本特征提取
    #1.获取数据
    data = ['life is short ,i like python','life is too long,i dislike python']
    #2. 文本特征转换
    #2.1 实例化
    transfer = CountVectorizer(stop_words=['dislike'])#没有sparse参数,不想分析的次写入stop_word
    #2.2 转换
    new_data = transfer.fit_transform(data)
    print(new_data.toarray())#将sparse矩阵取消需要使用.toarray()
    names = transfer.get_feature_names()
    print('属性名字是：\n',names)

def world_cut(text):
    #中文分词
    ret =' '.join(list(jieba.cut(text)))
    return ret

def chinese_demo():
    #中文文本特征提取
    #1.获取数据
    data = ['《小王子》是法国作家安托万·德·圣·埃克苏佩里于1942年写成的著名儿童文学短篇小说。本书的主人公是来自外星球',
            '的小王子。书中以一位飞行员作为故事叙述者，讲述了小王子从自己星球出发前往地球的过程中，所经历的各种历险。']
    #2,文章分割
    list = []
    for temp in data:
        list.append(world_cut(temp))
    print(list)
    #3. 文本特征转换
    #2.1 实例化
    transfer = CountVectorizer()#没有sparse参数,不想分析的次写入stop_word
    #2.2 转换
    new_data = transfer.fit_transform(list)
    print(new_data.toarray())#将sparse矩阵取消需要使用.toarray()
    names = transfer.get_feature_names()
    print('属性名字是：\n',names)

def tfidf_demo():
    #英文文本特征提取
    #1.获取数据
    data = ['《小王子》是法国作家安托万·德·圣·埃克苏佩里于1942年写成的著名儿童文学短篇小说。本书的主人公是来自外星球',
            '的小王子。书中以一位飞行员作为故事叙述者，讲述了小王子从自己星球出发前往地球的过程中，所经历的各种历险。']
    #2,文章分割
    list = []
    for temp in data:
        list.append(world_cut(temp))
    print(list)
    #3. 文本特征转换
    #2.1 实例化
    transfer = TfidfVectorizer()#词频和逆向文档频率
    #2.2 转换
    new_data = transfer.fit_transform(list)
    print(new_data.toarray())#将sparse矩阵取消需要使用.toarray()
    names = transfer.get_feature_names()
    print('属性名字是：\n',names)

# dict_demo()
# englis_demo()
chinese_demo()
tfidf_demo()