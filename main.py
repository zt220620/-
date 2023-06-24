import re
import warnings
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split

warnings.filterwarnings('ignore')

path = r"D:/for intstall/结果目录/design/"

# 导入数据集
labeled_data = pd.read_csv(path + 'labeled_data.csv')
unlabeled_data = pd.read_csv(path + 'unlabeled_data.csv')
test_data = pd.read_csv(path + 'test_data.csv')


labeled_data.head()

""" 创建分级信息映射 """
def rank_label(class_label):
    if class_label == '家居' or class_label == '体育' or class_label == '娱乐':
        return '可公开'
    elif class_label == '教育' or class_label == '时尚' or class_label == '游戏':
        return '低风险'
    elif class_label == '房产' or class_label == '科技':
        return '中风险'
    elif class_label == '财经' or class_label == '时政':
        return '高风险'
    
labeled_data['rank_label'] = labeled_data['class_label'].apply(rank_label)


""" 统计class_label各类的统计个数 """
labeled_data.class_label.value_counts()

def label_renum(class_label):
    if class_label == '房产':
        return 1
    elif class_label == '时政':
        return 2
    elif class_label == '家居':
        return 3
    elif class_label == '时尚':
        return 4
    elif class_label == '财经':
        return 5
    elif class_label == '教育':
        return 6
    else:
        return 7
    
labeled_data['class_label'] = labeled_data['class_label'].apply(label_renum)

def rank_score(rank_label):
    if rank_label == '可公开':
        return 1
    elif rank_label == '低风险':
        return 2
    elif rank_label == '中风险':
        return 3
    elif rank_label == '高风险':
        return 4

labeled_data['rank_label'] = labeled_data['rank_label'].apply(rank_score)

""" 删除数字和字母，以及标点符号 """
def re_sub(content):
    return re.sub('[0-9a-zA-Z~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）【】——+-=“：’；、。，？》《{}]+', ' ', content)

labeled_data['content'] = labeled_data['content'].apply(re_sub)
test_data['content'] = test_data['content'].apply(re_sub)


""" 去停用词表 """
stopwords = [line.strip() for line in open(path + 'baidu_stopwords.txt',encoding='UTF-8').readlines()]


""" 中文分词与停用词 """
def content_lcut(content):
    words = jieba.lcut(content)
    outstr = ''
    for word in words:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return list(filter(None, outstr.split(' ')))

labeled_data['content'] = labeled_data['content'].apply(content_lcut)
test_data['content'] = test_data['content'].apply(content_lcut)

""" 列表转为字符串 """
def list_string(content):
    return ','.join(content)

labeled_data['content'] = labeled_data['content'].apply(list_string)
test_data['content'] = test_data['content'].apply(list_string)


""" 计算TF-IDF  """
tf_transformer = TfidfVectorizer(stop_words='english', decode_error='ignore').fit(labeled_data.content)
x_train_counts_tf = tf_transformer.transform(labeled_data.content)
x_test_counts_tf = tf_transformer.transform(test_data.content)

print(x_train_counts_tf.shape)
print(x_test_counts_tf.shape)

""" 评估算法 """
num_folds = 10
seed = 7
scoring = 'accuracy'

# 生成算法模型
models = {}
models['SVM'] = SVC()


results = []
for key in models:
    kfold = KFold(n_splits=num_folds, random_state=seed,shuffle=True)
    cv_results = cross_val_score(models[key], x_train_counts_tf, labeled_data.class_label, cv=kfold, scoring=scoring)
    results.append(cv_results)
    print("%s : %f (%f)" % (key, cv_results.mean(), cv_results.std()))


    """ 划分训练集与测试集 """
x_train, x_test, y_train, y_test = train_test_split(x_train_counts_tf, labeled_data.class_label, test_size=0.3, random_state=100)

model = SVC().fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

""" 预测test_data数据集 """
y = model.predict(x_test_counts_tf)

test_data['class_label'] = y

def num_relabel(class_label):
    if class_label == 1:
        return '房产'
    elif class_label == 2:
        return '时政'
    elif class_label == 3:
        return '家居'
    elif class_label == 4:
        return '时尚'
    elif class_label == 5:
        return '财经'
    elif class_label == 6:
        return '教育'
    else:
        return '游戏'
    
test_data['class_label'] = test_data['class_label'].apply(num_relabel)

test_data['rank_label'] = test_data['class_label'].apply(rank_label)

data = test_data[['id', 'class_label', 'rank_label']]
data.to_csv(r"D:\\for intstall\\结果目录\\design\\result.csv", index=False)


