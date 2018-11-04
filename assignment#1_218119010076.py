# 导入pandas读csv
import pandas as pd
import matplotlib as mt
import matplotlib.pyplot as plt
import numpy as np

# 读入数据集

read_data = pd.read_csv('chinadep-soda2018.csv')
type(read_data)  # 类型为dataframe
# 检查数据集
read_data.dtypes

read_data.shape

# 替换性别分类为男：0，女：1
for i in range(1, len(read_data['性别推测'])):
    read_data['性别推测'] = read_data['性别推测'].replace('男', '0')
    read_data['性别推测'] = read_data['性别推测'].replace('女', '1')


# read_data = read_data.drop(['性别推测'], axis=1)


# 创建一个分类编码函数
def coding(col, codeDict):
    colCoded = pd.Series(col, copy=True)

    for key, value in codeDict.items():
        colCoded.replace(key, value, inplace=True)

    return colCoded


read_data['学历推测'] = coding(read_data['学历推测'], {'大学及以上': 4, '高中及以下': 3, '大学': 2, '高中': 1, '初中': 0})
read_data.学历推测.value_counts()
# read_data.rename(columns={'学历推测': 'education'}, inplace=True)

# 检查年龄段
read_data.年龄段推测.value_counts()

# 对年龄段分类
read_data['年龄段推测'] = coding(read_data['年龄段推测'], {'24岁以下': 0, '31-35岁': 1, '25-30岁': 2, '41岁以上': 3, '36-40岁': 4})
read_data.年龄段推测.value_counts()
# read_data.rename(columns={'年龄段推测': 'age'}, inplace=True)

# 检查是否有孩字段
read_data.是否有孩.value_counts()

# 对是否有孩分类
read_data['是否有孩'] = coding(read_data['是否有孩'], {'无孩子': 0, '有孩子': 1})
read_data.是否有孩.value_counts()
# read_data.rename(columns={'是否有孩': 'child'}, inplace=True)

# 检查婚姻状况字段
read_data.婚姻状况.value_counts()

# 对婚姻状况分类
read_data['婚姻状况'] = coding(read_data['婚姻状况'], {'未婚': 0, '已婚': 1})
read_data.婚姻状况.value_counts()
# read_data.rename(columns={'婚姻状况': 'marrige'}, inplace=True)
