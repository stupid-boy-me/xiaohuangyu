import pandas as pd
import time
from sklearn.feature_extraction import  DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV #网格搜索
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree


df = pd.read_csv(r"你的csv路径")
# print(df.info()) # RangeIndex: 45211 entries, 0 to 45210
# 观察数据集信息
# print(df.head())
# 观察数据集中是否存在重复数据
# print(df['age;'].count())
'''
第一步：数据处理 数据转变   看main函数
'''
'''
第二步：数据处理 观察数据集中是否存在缺失数据
'''
#print(df.isnull().sum())
# print(df["age"].describe())
# 我们取所有特征做分析演示，分别是：
x = df[['age', 'job', 'marital','education',
        'default','balance','housing','loan',
        'contact','day','month','duration',
        'campaign','pdays','previous','poutcome',]]

'''
第三步：特征抽取-onehot编码
'''
x_dict_list = x.to_dict(orient='records')
# print("*" * 30 + " train_dict " + "*" * 30)
# print(pd.Series(x_dict_list[:5]))
# 将类别转换成了one-hot编码
dict_vec = DictVectorizer(sparse=False)
x = dict_vec.fit_transform(x_dict_list)
print("*" * 30 + " onehot编码 " + "*" * 30)
print("*" * 30 + " 编码成功 " + "*" * 30)
# print(dict_vec.get_feature_names())
# print(x[:5])


y = df['y']
'''
第四步：划分训练集和测试集
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
'''
第五步：model-1-决策树分类器
'''
dec_tree = tree.DecisionTreeClassifier()
dec_tree.fit(x_train, y_train)
print("model-1-决策树分类器")
print("*" * 30 + " 准确率 " + "*" * 30)
print(dec_tree.score(x_test, y_test))

'''
第五步：model-2-KNN
'''
knn = KNeighborsClassifier(n_neighbors=2)    #实例化KNN模型
knn.fit(x_train, y_train)      #放入训练数据进行训练
#评估模型的得分
score=knn.score(x_test,y_test)
print("model-2-KNN")
print(score)

'''
第五步：model-3-随机森林分类器
       n_jobs: -1表示设置为核心数量
       n_estimators: 决策树数目
       max_depth: 树最大深度
'''
rf = RandomForestClassifier(n_jobs=-1)
param = {
    "n_estimators": [120, 200, 300, 500, 800, 1200],
    "max_depth": [5, 8, 15, 25, 30]
}
# 2折交叉验证
search = GridSearchCV(rf, param_grid=param, cv=2)
print("model-3-随机森林分类器")
print("*" * 30 + " 超参数网格搜索 " + "*" * 30)

start_time = time.time()
search.fit(x_train, y_train)
print("耗时：{}".format(time.time() - start_time))
print("最优参数：{}".format(search.best_params_))

print("*" * 30 + " 准确率 " + "*" * 30)
print(search.score(x_test, y_test))
