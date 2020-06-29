#-*-coding:utf-8-*-
'''
基于用户的协同过滤方法实现：

    数据集：ml-1m 
    步骤：
         在data-split.py中读取rating.dat的评分数据，生成训练集与测试集
        1 .读取训练集ml-train.txt
        2. 计算用户相似度
        3. 计算用户邻域
        4. 计算用户候选节目集，生成推荐列表
        读取测试集：
        6. 计算准确率，召回率
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math

train_file = './ml_train.txt'

# 1. 读取训练集ml-train.txt
#  生成获取用户-电影的评分矩阵
trainRatingsDF = pd.read_csv(train_file,sep='\t',names=['userId','movieId','rating'])
trainRatingsPivotDF = pd.pivot_table(trainRatingsDF[['userId','movieId','rating']],columns=['movieId'],index=['userId'],values='rating',fill_value=0)
ratingValues = trainRatingsPivotDF.values.tolist()  #(6040, 3681)


moviesMap = dict(enumerate(list(trainRatingsPivotDF.columns)))
usersMap = dict(enumerate(list(trainRatingsPivotDF.index)))

print(len(ratingValues))
# 2. 计算用户之间相似度
# 余弦相似度公式
def calCosineSimilarity(list1,list2):
    '''
        该算法是数学上的余弦相似度公式。
        根据《推荐系统实践》中第八章的公式，还需要计算每个用户的平均分，将用户的评分减去平均分，再进行计算。
    因为每个用户的评分范围不一致，减去平均分，有点类似归一化的感觉。
    '''
    res = 0
    denominator1 = 0
    denominator2 = 0
    for (val1,val2) in zip(list1,list2):
        res += (val1 * val2)
        denominator1 += val1 ** 2
        denominator2 += val2 ** 2
    return res / (math.sqrt(denominator1 * denominator2))

#计算用户相似度矩阵
#对于用户相似度矩阵，这是一个对称矩阵，同时对角线的元素为0，所以我们只需要计算上三角矩阵的值即可
user_sim_matrix = np.zeros((len(ratingValues), len(ratingValues)), dtype=np.float32)    #(6040, 6040)
for i in range(len(ratingValues) - 1):
    print("计算用户相似度矩阵: i=",i)
    for j in range(i + 1, len(ratingValues)):
        user_sim_matrix[i, j] = calCosineSimilarity(ratingValues[i], ratingValues[j])
        user_sim_matrix[j, i] = user_sim_matrix[i, j]
    print('\n')


# 3. 计算用户邻域
#  即找到与每个用户最相近的K个用户,这里K=10
user_most_sim_dict = dict() # (6040) ,{0:[userId,[]] ,....}
for i in range(len(ratingValues)):
    user_most_sim_dict[i] = sorted(enumerate(list(user_sim_matrix[i])),key = lambda x:x[1],reverse=True)[:10]   #通过enumerate，将值和下标变成键值对，下标就是用户编号
    print("找到与每个用户最相近的K个用户: i=",i,",",user_most_sim_dict[i])
    break

# 4. 生成用户候选集，生成推荐列表
# 预测用户-评分矩阵中未知的用户评分记录。
user_recommend_values = np.zeros((len(ratingValues),len(ratingValues[0])),dtype=np.float32)   #(6040, 3681)
for i in range(len(ratingValues)):
    for j in range(len(ratingValues[i])):
        if ratingValues[i][j] == 0:
            val = 0
            for (user, sim) in user_most_sim_dict[i]:
                val += (sim * ratingValues[user][j])
                user_recommend_values[i, j] = val
        print("预测用户-评分矩阵中未知的用户评分记录:i=",i,", j= ",j)
        print('\n')

# 5. 为用户推荐10电影
user_recommend_dict = dict()
user_recommend = './ml_train.txt'
fw_recommend = open(user_recommend, 'w')
for i in range(len(ratingValues)):
    user_recommend_dict[i] = sorted(enumerate(list(user_recommend_values[i])),key = lambda x:x[1],reverse=True)[:10]
    print("为用户推荐10电影: i=",i," ,user_recommend_dict=",user_recommend_dict[i])
    print('\n')

    line = i + '\t' + user_recommend_dict[i] + '\t' + '\n'
    fw_recommend.write(line)
fw_recommend.close()

# 6. 计算准确率
#读取测试集，生成 {user:items}对象
test_file = './ml_test.txt'
fr_test = open(test_file,'r')
test_dict = {}
for line in fr_test:
    lines = line.split('\t')
    user = lines[0]
    movie = lines[1]
    rating = lines[2].replace('\n', '')
    if user in test_dict:
        test_dict[user].append(movie)
    else :
        test_dict.update( {user:movie} )
fr_test.close()
# 计算准确率
user_size = 0
precision = 0.0
recall = 0.0
for user in test_dict:
    user_size += 1
    recommend_list = user_recommend_dict[user]
    test_list = test_dict[user]
    min_len = min(len(recommend_list), len(test_list))
    hit = 0
    for i in range(min_len):
        if recommend_list[i] in test_list:
            hit += 1
    # hit_ratio = float(hit / min_len)
    precision += float(hit / len(recommend_list))
    recall += float(hit / len(test_list))
precision = precision / user_size
recall = recall / user_size
print('The precision is : ',precision)
print('The recall is : ',recall)