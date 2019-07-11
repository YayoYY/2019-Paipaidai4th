'''Feature is being extracted in this file'''

import pandas as pd
import numpy as np
from config import *
from functions import *
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')

'''
1. reading
2. labeling
3. training set featuring
4. testing set featuring1

'''

### reading
print('reading...')

## 数据文件
str_names = ['test', 'train', 'listing_info', 'user_info', 'user_info_last_date', 'user_behavior_logs', 'user_taglist', 'user_taglist_last_date', 'user_repay_logs']
for name in str_names:1
    locals()['df_'+name] = pd.read_csv(path_data+locals()['file_'+name])
    print(name, 'data is ready...')

## 标签文件
with open(path_data+file_tags, 'r') as f:
    lst_tags = json.load(f)

## 预处理
df_train.replace({r'\N':np.nan}, inplace=True) # 缺失值处理
df_user_info.replace({'男': 'male', '女':'female'}, inplace=True) # 性别变为英文
df_user_info_last_date.replace({'男': 'male', '女':'female'}, inplace=True)
df_user_repay_logs['is_repay'] = df_user_repay_logs['repay_date'].apply(lambda x: 0 if x == '2200-01-01' else 1) # 是否逾期
df_user_repay_logs['repay_date'] = df_user_repay_logs['due_date'] + ':' + df_user_repay_logs['repay_date']
df_user_repay_logs['repay_date'] = df_user_repay_logs['repay_date'].apply(get_repay_due_gap) # 计算还款日期距离最晚还款日的间隔
df_user_taglist_last_date.replace({r'\N':'NULL'}, inplace=True) # 标签缺失值为NULL

### labeling
print('labeling...')

## 是否还款
df_train['is_repay'] = df_train['repay_date'].apply(lambda x: 0 if x is np.nan else 1) # 是否逾期

## 还款间隔
df_train['repay_date'] = df_train['due_date'] + ':' + df_train['repay_date']
df_train['repay_date'] = df_train['repay_date'].apply(get_repay_due_gap) # 计算还款日期距离最晚还款日的间隔

### training set featuring
print('training set featuring...')

## 用户静态维度(1.user_age 2.user_gender 3.user_id_province 4.user_cell_province)

# 用户性别
user_gender = df_user_info_last_date[df_user_info_last_date.user_id.isin(df_train.user_id.unique())][['user_id', 'gender']]
dummies_gender = pd.get_dummies(user_gender.gender, prefix='gender')
user_gender = pd.concat([user_gender, dummies_gender], axis=1)

# 用户性别-统计数据
gender_stat = pd.merge(df_user_repay_logs[df_user_repay_logs['is_repay'] == 1][['user_id', 'repay_date']], df_user_info_last_date[['user_id', 'gender']], on='user_id', how='left')
user_gender_gap_stat = get_stat(gender_stat, 'gender', 'repay_date', 'gender_repay_gap')
user_gender = pd.merge(user_gender, user_gender_gap_stat, left_on='gender', right_index=True)

# 用户性别-还款率
gender_ratio = pd.merge(df_user_repay_logs[['user_id', 'is_repay']], df_user_info_last_date[['user_id', 'gender']], on='user_id', how='left')
user_gender_repay_ratio = get_ratio(gender_ratio, 'gender', 'is_repay', 'gender_rey_ratio')
user_gender = pd.merge(user_gender, user_gender_repay_ratio, left_on='gender', right_index=True)

# 删除无用列
user_gender.drop(columns=['gender'], inplace=True) 

# 用户年龄
user_age = df_user_info_last_date[df_user_info_last_date.user_id.isin(df_train.user_id.unique())][['user_id', 'age']]
dummies_age = pd.get_dummies(user_age.age, prefix='age')
user_age = pd.concat([user_age, dummies_age], axis=1)

# 用户年龄-还款间隔统计数据
age_stat = pd.merge(df_user_repay_logs[df_user_repay_logs['is_repay'] == 1][['user_id', 'repay_date']], df_user_info_last_date[['user_id', 'age']], on='user_id', how='left')
user_age_gap_stat = get_stat(age_stat, 'age', 'repay_date', 'age_repay_gap')
user_age = pd.merge(user_age, user_age_gap_stat, left_on='age', right_index=True)

# 用户年龄-还款平均间隔排名
age_mean_gap_rank = user_age[['age', 'age_repay_gap_mean']].drop_duplicates().sort_values(by='age_repay_gap_mean')
age_mean_gap_rank['age_mean_gap_rank'] = [x for x in range(1, age_mean_gap_rank.shape[0]+1)]
age_mean_gap_rank.drop(columns=['age_repay_gap_mean'], inplace=True)
user_age = pd.merge(user_age, age_mean_gap_rank, on='age', how='left')

# 用户年龄-还款率
age_ratio = pd.merge(df_user_repay_logs[['user_id', 'is_repay']], df_user_info_last_date[['user_id', 'age']], on='user_id', how='left')
user_age_repay_ratio = get_ratio(age_ratio, 'age', 'is_repay', 'age_repay_ratio')
user_age = pd.merge(user_age, user_age_repay_ratio, left_on='age', right_index=True)

# 用户年龄-还款率排名
age_ratio_rank = user_age[['age', 'age_repay_ratio']].drop_duplicates().sort_values(by='age_repay_ratio')
age_ratio_rank['age_ratio_rank'] = [x for x in reversed(range(1, age_ratio_rank.shape[0]+1))]
age_ratio_rank.drop(columns=['age_repay_ratio'], inplace=True)
user_age = pd.merge(user_age, age_ratio_rank, on='age', how='left')

# 用户省份
user_id_province = df_user_info_last_date[df_user_info_last_date.user_id.isin(df_train.user_id.unique())][['user_id', 'id_province']]
dummies_id_province = pd.get_dummies(user_id_province.id_province, prefix='id_province')
user_id_province = pd.concat([user_id_province, dummies_id_province], axis=1)

# 用户省份-还款间隔统计数据
id_province_stat = pd.merge(df_user_repay_logs[df_user_repay_logs['is_repay'] == 1][['user_id', 'repay_date']], df_user_info_last_date[['user_id', 'id_province']], on='user_id', how='left')
user_id_province_gap_stat = get_stat(id_province_stat, 'id_province', 'repay_date', 'id_province_repay_gap')
user_id_province = pd.merge(user_id_province, user_id_province_gap_stat, left_on='id_province', right_index=True)

# 用户省份-还款平均间隔排名
id_province_mean_gap_rank = user_id_province[['id_province', 'id_province_repay_gap_mean']].drop_duplicates().sort_values(by='id_province_repay_gap_mean')
id_province_mean_gap_rank['id_province_mean_gap_rank'] = [x for x in range(1, id_province_mean_gap_rank.shape[0]+1)]
id_province_mean_gap_rank.drop(columns=['id_province_repay_gap_mean'], inplace=True)
user_id_province = pd.merge(user_id_province, id_province_mean_gap_rank, on='id_province', how='left')

# 用户省份-还款率
id_province_ratio = pd.merge(df_user_repay_logs[['user_id', 'is_repay']], df_user_info_last_date[['user_id', 'id_province']], on='user_id', how='left')
user_id_province_repay_ratio = get_ratio(id_province_ratio, 'id_province', 'is_repay', 'id_province_repay_ratio')
user_id_province = pd.merge(user_id_province, user_id_province_repay_ratio, left_on='id_province', right_index=True)

# 用户省份-还款率排名
id_province_ratio_rank = user_id_province[['id_province', 'id_province_repay_ratio']].drop_duplicates().sort_values(by='id_province_repay_ratio')
id_province_ratio_rank['id_province_ratio_rank'] = [x for x in reversed(range(1, id_province_ratio_rank.shape[0]+1))]
id_province_ratio_rank.drop(columns=['id_province_repay_ratio'], inplace=True)
user_id_province = pd.merge(user_id_province, id_province_ratio_rank, on='id_province', how='left')

# 用户省份-删除无用列
user_id_province.drop(columns=['id_province'], inplace=True)

# 用户手机省份
user_cell_province = df_user_info_last_date[df_user_info_last_date.user_id.isin(df_train.user_id.unique())][['user_id', 'cell_province']]
dummies_cell_province = pd.get_dummies(user_cell_province.cell_province, prefix='cell_province')
user_cell_province = pd.concat([user_cell_province, dummies_cell_province], axis=1)

# 用户手机省份-还款间隔统计数据
cell_province_stat = pd.merge(df_user_repay_logs[df_user_repay_logs['is_repay'] == 1][['user_id', 'repay_date']], df_user_info_last_date[['user_id', 'cell_province']], on='user_id', how='left')
user_cell_province_gap_stat = get_stat(cell_province_stat, 'cell_province', 'repay_date', 'cell_province_repay_gap')
user_cell_province = pd.merge(user_cell_province, user_cell_province_gap_stat, left_on='cell_province', right_index=True)

# 用户手机省份-还款平均间隔排名
cell_province_mean_gap_rank = user_cell_province[['cell_province', 'cell_province_repay_gap_mean']].drop_duplicates().sort_values(by='cell_province_repay_gap_mean')
cell_province_mean_gap_rank['cell_province_mean_gap_rank'] = [x for x in range(1, cell_province_mean_gap_rank.shape[0]+1)]
cell_province_mean_gap_rank.drop(columns=['cell_province_repay_gap_mean'], inplace=True)
user_cell_province = pd.merge(user_cell_province, cell_province_mean_gap_rank, on='cell_province', how='left')

# 用户手机省份-还款率
cell_province_ratio = pd.merge(df_user_repay_logs[['user_id', 'is_repay']], df_user_info_last_date[['user_id', 'cell_province']], on='user_id', how='left')
user_cell_province_repay_ratio = get_ratio(cell_province_ratio, 'cell_province', 'is_repay', 'cell_province_repay_ratio')
user_cell_province = pd.merge(user_cell_province, user_cell_province_repay_ratio, left_on='cell_province', right_index=True)

# 用户手机省份-还款率排名
cell_province_ratio_rank = user_cell_province[['cell_province', 'cell_province_repay_ratio']].drop_duplicates().sort_values(by='cell_province_repay_ratio')
cell_province_ratio_rank['cell_province_ratio_rank'] = [x for x in reversed(range(1, cell_province_ratio_rank.shape[0]+1))]
cell_province_ratio_rank.drop(columns=['cell_province_repay_ratio'], inplace=True)
user_cell_province = pd.merge(user_cell_province, cell_province_ratio_rank, on='cell_province', how='left')

# 用户手机省份-删除无用列
user_cell_province.drop(columns=['cell_province'], inplace=True)

## 用户动态维度(1.user_basic 2.user_beha 3.user_tag 4. user_repay)

# 用户基础信息
user_basic = df_user_info[df_user_info.user_id.isin(df_train.user_id.unique())][['user_id', 'insertdate']]

# 用户基础信息-更换次数
user_basic_count = user_basic.groupby('user_id').agg(lambda x: len(x)).reset_index()
user_basic_count.rename(columns={'insertdate': 'user_basic_count'}, inplace=True)
user_basic = pd.merge(user_basic, user_basic_count, on='user_id', how='left')

# 用户基础信息-更换频率
user_basic['insertdate'] = user_basic['insertdate'].apply(get_time_string)
user_basic_freq = user_basic.groupby('user_id')['insertdate'].agg(lambda x:':'.join(x)).reset_index()
user_basic_freq['insertdate'] = user_basic_freq['insertdate'].apply(get_max_time_gap)
user_basic_freq.rename(columns={'insertdate': 'insert_interval'}, inplace=True)
user_basic = pd.merge(user_basic, user_basic_freq, on='user_id', how='left')
user_basic['insert_interval'] = user_basic['insert_interval'].replace({0:np.nan})
user_basic['user_basic_update_freq'] = user_basic['user_basic_count'] / user_basic['insert_interval']
user_basic.drop(columns=['insert_interval'], inplace=True)
user_basic.fillna(0, inplace=True)

# 用户基础信息-删除无用列
user_basic.drop(columns='insertdate', inplace=True)

# 用户基础信息-去掉重复值
user_basic.drop_duplicates(inplace=True)

# 用户操作
user_beha = df_user_behavior_logs[df_user_behavior_logs.user_id.isin(df_train.user_id.unique())].copy()

# 用户操作-记录数
user_beha_count = user_beha[['user_id']]
user_beha_count['user_beha_count'] = 1 
user_beha_count = user_beha_count.groupby('user_id').agg(sum).reset_index()

# 用户操作-最多/少的行为类别
user_beha_type = user_beha.groupby(['user_id', 'behavior_type']).agg(lambda x:len(x)).reset_index()
user_beha_type['tmp'] = user_beha_type['behavior_type'].astype(str) + ':' + user_beha_type['behavior_time'].astype(str)
user_beha_type = user_beha_type.groupby('user_id')['tmp'].agg(lambda x:'+'.join(x)).reset_index()
user_beha_type['user_beha_max_type'] = user_beha_type['tmp'].apply(get_most_beha, args=('max', ))
user_beha_type['user_beha_min_type'] = user_beha_type['tmp'].apply(get_most_beha, args=('min', ))
dummies_max_beha = pd.get_dummies(user_beha_type.user_beha_max_type, prefix='user_beha_max_type')
user_beha_type = pd.concat([user_beha_type, dummies_max_beha], axis=1)
dummies_min_beha = pd.get_dummies(user_beha_type.user_beha_min_type, prefix='user_beha_min_type')
user_beha_type = pd.concat([user_beha_type, dummies_min_beha], axis=1)
user_beha_type.drop(columns=['tmp', 'user_beha_max_type', 'user_beha_min_type'], inplace=True)

# 用户操作-发生行为1的数量
user_beha_1_count = user_beha[user_beha.behavior_type == 1][['user_id']]
user_beha_1_count['user_beha_1_count'] = 1
user_beha_1_count = user_beha_1_count.groupby('user_id').agg(sum).reset_index()

# 用户操作-发生行为2的数量
user_beha_2_count = user_beha[user_beha.behavior_type == 2][['user_id']]
user_beha_2_count['user_beha_2_count'] = 1
user_beha_2_count = user_beha_2_count.groupby('user_id').agg(sum).reset_index()

# 用户操作-发生行为3的数量
user_beha_3_count = user_beha[user_beha.behavior_type == 3][['user_id']]
user_beha_3_count['user_beha_3_count'] = 1
user_beha_3_count = user_beha_3_count.groupby('user_id').agg(sum).reset_index()

# 用户操作-去掉重复值
user_basic.drop_duplicates(inplace=True)

# 用户操作-合并
user_beha = user_beha[['user_id']].drop_duplicates()
user_beha = pd.merge(user_beha, user_beha_count, on='user_id', how='left')
user_beha = pd.merge(user_beha, user_beha_type, on='user_id', how='left')
user_beha = pd.merge(user_beha, user_beha_1_count, on='user_id', how='left')
user_beha = pd.merge(user_beha, user_beha_2_count, on='user_id', how='left')
user_beha = pd.merge(user_beha, user_beha_3_count, on='user_id', how='left')

# 用户标签-更改次数
user_tag_count = df_user_taglist[df_user_taglist.user_id.isin(df_train.user_id.unique())][['user_id']]
user_tag_count['user_tag_count'] = 1
user_tag_count = user_tag_count.groupby('user_id').agg(sum).reset_index()

# 用户标签数-统计
user_tag = df_user_taglist_last_date[df_user_taglist_last_date.user_id.isin(df_train.user_id.unique())][['user_id', 'taglist']]
all_tags = ''
for tag in user_tag.taglist:
    all_tags += tag + '|'
all_tags = all_tags.split('|')[:-2]
tag_count = Counter()
for tag in all_tags:
    tag_count[tag] = tag_count[tag] + 1
tag_count = dict(tag_count)

sort_tag_count_des = sorted(tag_count.items(),key=lambda item:item[1],reverse=True)
most_100_tags = [sort_tag_count_des[i][0] for i in range(100)] # 最多的十个标签
sort_tag_count_asc = sorted(tag_count.items(),key=lambda item:item[1],reverse=False)
least_100_tags = [sort_tag_count_asc[i][0] for i in range(100)] # 最少的十个标签

# 用户标签-占有多少个100个热门标签
user_tag['most_100'] = 0
for tag in most_100_tags:
    user_tag.loc[user_tag.taglist.str.contains(tag), 'most_100'] += 1

user_tag['least_100'] = 0
for tag in least_100_tags:
    user_tag.loc[user_tag.taglist.str.contains(tag), 'least_100'] += 1

# 用户标签-是否在100个热门/冷门标签中
user_tag['is_in_most_100'] = 0
for tag in most_100_tags:
    user_tag.loc[user_tag.taglist.str.contains(tag), 'is_in_most_100'] = 1
user_tag['is_in_least_100'] = 0
for tag in least_100_tags:
    user_tag.loc[user_tag.taglist.str.contains(tag), 'is_in_least_100'] = 1

# 用户标签-长度
user_tag_len = df_user_taglist_last_date[df_user_taglist_last_date.user_id.isin(df_train.user_id.unique())][['user_id', 'taglist']]
user_tag_len['taglist'] = user_tag_len['taglist'].apply(lambda x:len(x.split('|')))
user_tag_len.rename(columns={'taglist':'user_tag_len'}, inplace=True)

# 用户标签-事先合并
user_tag = pd.merge(user_tag, user_tag_count, on='user_id', how='left')
user_tag = pd.merge(user_tag, user_tag_len, on='user_id', how='left')

# 用户标签-所属标签平均还款率的平均/最高/最低/中位数
user_is_repay = df_user_repay_logs[df_user_repay_logs.user_id.isin(user_tag.user_id.unique())][['user_id', 'is_repay']]
user_is_repay = user_is_repay.groupby('user_id').agg(lambda x:sum(x)/len(x)).reset_index()
user_tag = pd.merge(user_tag, user_is_repay, on='user_id', how='left')
user_tag['taglist'] = '|' + user_tag['taglist'] + '|'
# tag_ratio = {}
# for i, tag in enumerate(tag_count.keys()):
#     tag_ratio[tag] = user_tag[user_tag.taglist.str.contains('\|'+tag+'\|')]['is_repay'].sum()
#     tag_ratio[tag] = tag_ratio[tag] / tag_count[tag]
with open(path_data+file_tag_ratio, 'r') as f:
    tag_ratio = json.load(f)
for colname in ['_mean', '_max', '_min', '_50']:
    user_tag['tag_ratio'+colname] = user_tag.taglist.apply(get_stat_1, args=(tag_ratio, colname))    

# 用户标签-删除无用列
user_tag.drop(columns=['taglist', 'is_repay'], inplace=True)

# 用户借款-记录数
user_repay_count = df_user_repay_logs[df_user_repay_logs.user_id.isin(df_train.user_id.unique())][['user_id']]
user_repay_count['user_repay_count'] = 1
user_repay_count = user_repay_count.groupby('user_id').agg(sum).reset_index()

# 用户借款-频率
user_repay_freq = df_user_repay_logs[df_user_repay_logs.user_id.isin(df_train.user_id.unique())][['user_id', 'due_date']]
user_repay_freq['due_date'] = user_repay_freq['due_date'].apply(get_time_string)
user_repay_freq = user_repay_freq.groupby('user_id').agg(lambda x:':'.join(x)).reset_index()
user_repay_freq['due_date'] = user_repay_freq['due_date'].apply(get_max_time_gap)
user_repay_freq.replace({0:np.nan}, inplace=True)
user_repay_freq['due_date'] = user_repay_count['user_repay_count'] / user_repay_freq['due_date']
user_repay_freq.rename(columns={'due_date':'user_repay_freq'}, inplace=True)

# 用户借款-最大/最小/平均时间间隔
user_repay_interval = df_user_repay_logs[df_user_repay_logs.user_id.isin(df_train.user_id.unique())][['user_id', 'due_date']]
user_repay_interval['due_date'] = user_repay_interval['due_date'].apply(get_time_string)
user_repay_interval = user_repay_interval.groupby('user_id').agg(lambda x:':'.join(x)).reset_index()
user_repay_interval['user_repay_interval_max'] = user_repay_interval['due_date'].apply(get_repay_interval_1, args=('max', ))
user_repay_interval['user_repay_interval_min'] = user_repay_interval['due_date'].apply(get_repay_interval_1, args=('min', ))
user_repay_interval['user_repay_interval_avg'] = user_repay_interval['due_date'].apply(get_repay_interval_1, args=('avg', ))

# 用户借款-期数统计数据
user_repay_order = df_user_repay_logs[df_user_repay_logs.user_id.isin(df_train.user_id.unique())][['user_id', 'order_id']]
user_repay_order = get_stat(user_repay_order, 'user_id', 'order_id', 'user_repay_order')
user_repay_order.drop(columns='user_repay_order_mode', inplace=True)

# 用户借款-还款金额统计数据
user_repay_amt = df_user_repay_logs[df_user_repay_logs.user_id.isin(df_train.user_id.unique())][['user_id', 'due_amt']]
user_repay_amt = get_stat(user_repay_amt, 'user_id', 'due_amt', 'user_repay_amt')
user_repay_amt.drop(columns='user_repay_amt_mode', inplace=True)

# 用户借款-还款间隔统计数据
user_repay_gap = df_user_repay_logs[(df_user_repay_logs.user_id.isin(df_train.user_id.unique())) & (df_user_repay_logs.repay_date > 0)][['user_id', 'repay_date']]
user_repay_gap = get_stat(user_repay_gap, 'user_id', 'repay_date', 'user_repay_gap')
user_repay_gap.drop(columns='user_repay_gap_mode', inplace=True)

# 用户借款-用户还款率
user_repay_ratio = df_user_repay_logs[df_user_repay_logs.user_id.isin(df_train.user_id.unique())][['user_id', 'is_repay']]
user_repay_ratio = user_repay_ratio.groupby('user_id').agg(lambda x:sum(x)/len(x)).reset_index()
user_repay_ratio.rename(columns={'is_repay':'user_repay_ratio'}, inplace=True)

# 用户借款-合并
user_repay = df_user_repay_logs[df_user_repay_logs.user_id.isin(df_train.user_id.unique())][['user_id']].drop_duplicates()
user_repay = pd.merge(user_repay, user_repay_count, on='user_id', how='left')
user_repay = pd.merge(user_repay, user_repay_freq, on='user_id', how='left')
user_repay = pd.merge(user_repay, user_repay_interval, on='user_id', how='left')
user_repay = pd.merge(user_repay, user_repay_ratio, on='user_id', how='left')
user_repay = pd.merge(user_repay, user_repay_order, left_on='user_id', right_index=True, how='left')
user_repay = pd.merge(user_repay, user_repay_amt, left_on='user_id', right_index=True, how='left')
user_repay = pd.merge(user_repay, user_repay_gap, left_on='user_id', right_index=True, how='left')
user_repay.drop(columns=['due_date'], inplace=True)

## 标签维度

listing = pd.merge(df_train[['listing_id']], df_listing_info[['listing_id', 'term', 'rate', 'principal']], on='listing_id', how='left')
listing['listing_profit'] = listing['principal'] * listing['rate']
listing['listing_repay'] = listing['listing_profit'] / listing['term']

## 备份
user_gender.to_csv(path_data+file_user_gender, index=None)
user_age.to_csv(path_data+file_user_age, index=None)
user_cell_province.to_csv(path_data+file_user_id_province, index=None)
user_id_province.to_csv(path_data+file_user_cell_province, index=None)
user_basic.to_csv(path_data+file_user_basic, index=None)
user_beha.to_csv(path_data+file_user_beha, index=None)
user_tag.to_csv(path_data+file_user_tag, index=None)
user_repay.to_csv(path_data+file_user_repay, index=None)
listing.to_csv(path_data+file_listing, index=None)

## 构造训练集特征
user = df_train[['user_id']].drop_duplicates()
user = pd.merge(user, user_gender, on='user_id', how='left')
user = pd.merge(user, user_age, on='user_id', how='left')
user = pd.merge(user, user_cell_province, on='user_id', how='left')
user = pd.merge(user, user_id_province, on='user_id', how='left')
user = pd.merge(user, user_basic, on='user_id', how='left')
user = pd.merge(user, user_beha, on='user_id', how='left')
user = pd.merge(user, user_tag, on='user_id', how='left')
user = pd.merge(user, user_repay, on='user_id', how='left')
df_train_feat = pd.merge(df_train, user, on='user_id', how='left')
df_train_feat = pd.merge(df_train_feat, listing, on='listing_id', how='left')
df_train_feat.drop(columns=['auditing_date', 'due_date', 'repay_amt'], inplace=True)
df_train_feat.to_csv(path_data+file_train_feat, index=None)

### testing set featuring
print('testing set featuring...')

## 用户静态维度(1.user_age 2.user_gender 3.user_id_province 4.user_cell_province)

# 用户性别
user_gender = df_user_info_last_date[df_user_info_last_date.user_id.isin(df_test.user_id.unique())][['user_id', 'gender']]
dummies_gender = pd.get_dummies(user_gender.gender, prefix='gender')
user_gender = pd.concat([user_gender, dummies_gender], axis=1)

# 用户性别-统计数据
gender_stat = pd.merge(df_user_repay_logs[df_user_repay_logs['is_repay'] == 1][['user_id', 'repay_date']], df_user_info_last_date[['user_id', 'gender']], on='user_id', how='left')
user_gender_gap_stat = get_stat(gender_stat, 'gender', 'repay_date', 'gender_repay_gap')
user_gender = pd.merge(user_gender, user_gender_gap_stat, left_on='gender', right_index=True)

# 用户性别-还款率
gender_ratio = pd.merge(df_user_repay_logs[['user_id', 'is_repay']], df_user_info_last_date[['user_id', 'gender']], on='user_id', how='left')
user_gender_repay_ratio = get_ratio(gender_ratio, 'gender', 'is_repay', 'gender_rey_ratio')
user_gender = pd.merge(user_gender, user_gender_repay_ratio, left_on='gender', right_index=True)

# 删除无用列
user_gender.drop(columns=['gender'], inplace=True) 

# 用户年龄
user_age = df_user_info_last_date[df_user_info_last_date.user_id.isin(df_test.user_id.unique())][['user_id', 'age']]
dummies_age = pd.get_dummies(user_age.age, prefix='age')
user_age = pd.concat([user_age, dummies_age], axis=1)

# 用户年龄-还款间隔统计数据
age_stat = pd.merge(df_user_repay_logs[df_user_repay_logs['is_repay'] == 1][['user_id', 'repay_date']], df_user_info_last_date[['user_id', 'age']], on='user_id', how='left')
user_age_gap_stat = get_stat(age_stat, 'age', 'repay_date', 'age_repay_gap')
user_age = pd.merge(user_age, user_age_gap_stat, left_on='age', right_index=True)

# 用户年龄-还款平均间隔排名
age_mean_gap_rank = user_age[['age', 'age_repay_gap_mean']].drop_duplicates().sort_values(by='age_repay_gap_mean')
age_mean_gap_rank['age_mean_gap_rank'] = [x for x in range(1, age_mean_gap_rank.shape[0]+1)]
age_mean_gap_rank.drop(columns=['age_repay_gap_mean'], inplace=True)
user_age = pd.merge(user_age, age_mean_gap_rank, on='age', how='left')

# 用户年龄-还款率
age_ratio = pd.merge(df_user_repay_logs[['user_id', 'is_repay']], df_user_info_last_date[['user_id', 'age']], on='user_id', how='left')
user_age_repay_ratio = get_ratio(age_ratio, 'age', 'is_repay', 'age_repay_ratio')
user_age = pd.merge(user_age, user_age_repay_ratio, left_on='age', right_index=True)

# 用户年龄-还款率排名
age_ratio_rank = user_age[['age', 'age_repay_ratio']].drop_duplicates().sort_values(by='age_repay_ratio')
age_ratio_rank['age_ratio_rank'] = [x for x in reversed(range(1, age_ratio_rank.shape[0]+1))]
age_ratio_rank.drop(columns=['age_repay_ratio'], inplace=True)
user_age = pd.merge(user_age, age_ratio_rank, on='age', how='left')

# 用户省份
user_id_province = df_user_info_last_date[df_user_info_last_date.user_id.isin(df_test.user_id.unique())][['user_id', 'id_province']]
dummies_id_province = pd.get_dummies(user_id_province.id_province, prefix='id_province')
user_id_province = pd.concat([user_id_province, dummies_id_province], axis=1)

# 用户省份-还款间隔统计数据
id_province_stat = pd.merge(df_user_repay_logs[df_user_repay_logs['is_repay'] == 1][['user_id', 'repay_date']], df_user_info_last_date[['user_id', 'id_province']], on='user_id', how='left')
user_id_province_gap_stat = get_stat(id_province_stat, 'id_province', 'repay_date', 'id_province_repay_gap')
user_id_province = pd.merge(user_id_province, user_id_province_gap_stat, left_on='id_province', right_index=True)

# 用户省份-还款平均间隔排名
id_province_mean_gap_rank = user_id_province[['id_province', 'id_province_repay_gap_mean']].drop_duplicates().sort_values(by='id_province_repay_gap_mean')
id_province_mean_gap_rank['id_province_mean_gap_rank'] = [x for x in range(1, id_province_mean_gap_rank.shape[0]+1)]
id_province_mean_gap_rank.drop(columns=['id_province_repay_gap_mean'], inplace=True)
user_id_province = pd.merge(user_id_province, id_province_mean_gap_rank, on='id_province', how='left')

# 用户省份-还款率
id_province_ratio = pd.merge(df_user_repay_logs[['user_id', 'is_repay']], df_user_info_last_date[['user_id', 'id_province']], on='user_id', how='left')
user_id_province_repay_ratio = get_ratio(id_province_ratio, 'id_province', 'is_repay', 'id_province_repay_ratio')
user_id_province = pd.merge(user_id_province, user_id_province_repay_ratio, left_on='id_province', right_index=True)

# 用户省份-还款率排名
id_province_ratio_rank = user_id_province[['id_province', 'id_province_repay_ratio']].drop_duplicates().sort_values(by='id_province_repay_ratio')
id_province_ratio_rank['id_province_ratio_rank'] = [x for x in reversed(range(1, id_province_ratio_rank.shape[0]+1))]
id_province_ratio_rank.drop(columns=['id_province_repay_ratio'], inplace=True)
user_id_province = pd.merge(user_id_province, id_province_ratio_rank, on='id_province', how='left')

# 用户省份-删除无用列
user_id_province.drop(columns=['id_province'], inplace=True)

# 用户手机省份
user_cell_province = df_user_info_last_date[df_user_info_last_date.user_id.isin(df_test.user_id.unique())][['user_id', 'cell_province']]
dummies_cell_province = pd.get_dummies(user_cell_province.cell_province, prefix='cell_province')
user_cell_province = pd.concat([user_cell_province, dummies_cell_province], axis=1)

# 用户手机省份-还款间隔统计数据
cell_province_stat = pd.merge(df_user_repay_logs[df_user_repay_logs['is_repay'] == 1][['user_id', 'repay_date']], df_user_info_last_date[['user_id', 'cell_province']], on='user_id', how='left')
user_cell_province_gap_stat = get_stat(cell_province_stat, 'cell_province', 'repay_date', 'cell_province_repay_gap')
user_cell_province = pd.merge(user_cell_province, user_cell_province_gap_stat, left_on='cell_province', right_index=True)

# 用户手机省份-还款平均间隔排名
cell_province_mean_gap_rank = user_cell_province[['cell_province', 'cell_province_repay_gap_mean']].drop_duplicates().sort_values(by='cell_province_repay_gap_mean')
cell_province_mean_gap_rank['cell_province_mean_gap_rank'] = [x for x in range(1, cell_province_mean_gap_rank.shape[0]+1)]
cell_province_mean_gap_rank.drop(columns=['cell_province_repay_gap_mean'], inplace=True)
user_cell_province = pd.merge(user_cell_province, cell_province_mean_gap_rank, on='cell_province', how='left')

# 用户手机省份-还款率
cell_province_ratio = pd.merge(df_user_repay_logs[['user_id', 'is_repay']], df_user_info_last_date[['user_id', 'cell_province']], on='user_id', how='left')
user_cell_province_repay_ratio = get_ratio(cell_province_ratio, 'cell_province', 'is_repay', 'cell_province_repay_ratio')
user_cell_province = pd.merge(user_cell_province, user_cell_province_repay_ratio, left_on='cell_province', right_index=True)

# 用户手机省份-还款率排名
cell_province_ratio_rank = user_cell_province[['cell_province', 'cell_province_repay_ratio']].drop_duplicates().sort_values(by='cell_province_repay_ratio')
cell_province_ratio_rank['cell_province_ratio_rank'] = [x for x in reversed(range(1, cell_province_ratio_rank.shape[0]+1))]
cell_province_ratio_rank.drop(columns=['cell_province_repay_ratio'], inplace=True)
user_cell_province = pd.merge(user_cell_province, cell_province_ratio_rank, on='cell_province', how='left')

# 用户手机省份-删除无用列
user_cell_province.drop(columns=['cell_province'], inplace=True)

## 用户动态维度(1.user_basic 2.user_beha 3.user_tag 4. user_repay)

# 用户基础信息
user_basic = df_user_info[df_user_info.user_id.isin(df_test.user_id.unique())][['user_id', 'insertdate']]

# 用户基础信息-更换次数
user_basic_count = user_basic.groupby('user_id').agg(lambda x: len(x)).reset_index()
user_basic_count.rename(columns={'insertdate': 'user_basic_count'}, inplace=True)
user_basic = pd.merge(user_basic, user_basic_count, on='user_id', how='left')

# 用户基础信息-更换频率
user_basic['insertdate'] = user_basic['insertdate'].apply(get_time_string)
user_basic_freq = user_basic.groupby('user_id')['insertdate'].agg(lambda x:':'.join(x)).reset_index()
user_basic_freq['insertdate'] = user_basic_freq['insertdate'].apply(get_max_time_gap)
user_basic_freq.rename(columns={'insertdate': 'insert_interval'}, inplace=True)
user_basic = pd.merge(user_basic, user_basic_freq, on='user_id', how='left')
user_basic['insert_interval'] = user_basic['insert_interval'].replace({0:np.nan})
user_basic['user_basic_update_freq'] = user_basic['user_basic_count'] / user_basic['insert_interval']
user_basic.drop(columns=['insert_interval'], inplace=True)
user_basic.fillna(0, inplace=True)

# 用户基础信息-删除无用列
user_basic.drop(columns='insertdate', inplace=True)

# 用户基础信息-去掉重复值
user_basic.drop_duplicates(inplace=True)

# 用户操作
user_beha = df_user_behavior_logs[df_user_behavior_logs.user_id.isin(df_test.user_id.unique())].copy()

# 用户操作-记录数
user_beha_count = user_beha[['user_id']]
user_beha_count['user_beha_count'] = 1 
user_beha_count = user_beha_count.groupby('user_id').agg(sum).reset_index()

# 用户操作-最多/少的行为类别
user_beha_type = user_beha.groupby(['user_id', 'behavior_type']).agg(lambda x:len(x)).reset_index()
user_beha_type['tmp'] = user_beha_type['behavior_type'].astype(str) + ':' + user_beha_type['behavior_time'].astype(str)
user_beha_type = user_beha_type.groupby('user_id')['tmp'].agg(lambda x:'+'.join(x)).reset_index()
user_beha_type['user_beha_max_type'] = user_beha_type['tmp'].apply(get_most_beha, args=('max', ))
user_beha_type['user_beha_min_type'] = user_beha_type['tmp'].apply(get_most_beha, args=('min', ))
dummies_max_beha = pd.get_dummies(user_beha_type.user_beha_max_type, prefix='user_beha_max_type')
user_beha_type = pd.concat([user_beha_type, dummies_max_beha], axis=1)
dummies_min_beha = pd.get_dummies(user_beha_type.user_beha_min_type, prefix='user_beha_min_type')
user_beha_type = pd.concat([user_beha_type, dummies_min_beha], axis=1)
user_beha_type.drop(columns=['tmp', 'user_beha_max_type', 'user_beha_min_type'], inplace=True)

# 用户操作-发生行为1的数量
user_beha_1_count = user_beha[user_beha.behavior_type == 1][['user_id']]
user_beha_1_count['user_beha_1_count'] = 1
user_beha_1_count = user_beha_1_count.groupby('user_id').agg(sum).reset_index()

# 用户操作-发生行为2的数量
user_beha_2_count = user_beha[user_beha.behavior_type == 2][['user_id']]
user_beha_2_count['user_beha_2_count'] = 1
user_beha_2_count = user_beha_2_count.groupby('user_id').agg(sum).reset_index()

# 用户操作-发生行为3的数量
user_beha_3_count = user_beha[user_beha.behavior_type == 3][['user_id']]
user_beha_3_count['user_beha_3_count'] = 1
user_beha_3_count = user_beha_3_count.groupby('user_id').agg(sum).reset_index()

# 用户操作-去掉重复值
user_basic.drop_duplicates(inplace=True)

# 用户操作-合并
user_beha = user_beha[['user_id']].drop_duplicates()
user_beha = pd.merge(user_beha, user_beha_count, on='user_id', how='left')
user_beha = pd.merge(user_beha, user_beha_type, on='user_id', how='left')
user_beha = pd.merge(user_beha, user_beha_1_count, on='user_id', how='left')
user_beha = pd.merge(user_beha, user_beha_2_count, on='user_id', how='left')
user_beha = pd.merge(user_beha, user_beha_3_count, on='user_id', how='left')

# 用户标签-更改次数
user_tag_count = df_user_taglist[df_user_taglist.user_id.isin(df_test.user_id.unique())][['user_id']]
user_tag_count['user_tag_count'] = 1
user_tag_count = user_tag_count.groupby('user_id').agg(sum).reset_index()

# 用户标签数-统计
user_tag = df_user_taglist_last_date[df_user_taglist_last_date.user_id.isin(df_test.user_id.unique())][['user_id', 'taglist']]
all_tags = ''
for tag in user_tag.taglist:
    all_tags += tag + '|'
all_tags = all_tags.split('|')[:-2]
tag_count = Counter()
for tag in all_tags:
    tag_count[tag] = tag_count[tag] + 1
tag_count = dict(tag_count)

sort_tag_count_des = sorted(tag_count.items(),key=lambda item:item[1],reverse=True)
most_100_tags = [sort_tag_count_des[i][0] for i in range(100)] # 最多的十个标签
sort_tag_count_asc = sorted(tag_count.items(),key=lambda item:item[1],reverse=False)
least_100_tags = [sort_tag_count_asc[i][0] for i in range(100)] # 最少的十个标签

# 用户标签-占有多少个100个热门标签
user_tag['most_100'] = 0
for tag in most_100_tags:
    user_tag.loc[user_tag.taglist.str.contains(tag), 'most_100'] += 1

user_tag['least_100'] = 0
for tag in least_100_tags:
    user_tag.loc[user_tag.taglist.str.contains(tag), 'least_100'] += 1

# 用户标签-是否在100个热门/冷门标签中
user_tag['is_in_most_100'] = 0
for tag in most_100_tags:
    user_tag.loc[user_tag.taglist.str.contains(tag), 'is_in_most_100'] = 1
user_tag['is_in_least_100'] = 0
for tag in least_100_tags:
    user_tag.loc[user_tag.taglist.str.contains(tag), 'is_in_least_100'] = 1

# 用户标签-长度
user_tag_len = df_user_taglist_last_date[df_user_taglist_last_date.user_id.isin(df_test.user_id.unique())][['user_id', 'taglist']]
user_tag_len['taglist'] = user_tag_len['taglist'].apply(lambda x:len(x.split('|')))
user_tag_len.rename(columns={'taglist':'user_tag_len'}, inplace=True)

# 用户标签-事先合并
user_tag = pd.merge(user_tag, user_tag_count, on='user_id', how='left')
user_tag = pd.merge(user_tag, user_tag_len, on='user_id', how='left')

# 用户标签-所属标签平均还款率的平均/最高/最低/中位数
user_is_repay = df_user_repay_logs[df_user_repay_logs.user_id.isin(user_tag.user_id.unique())][['user_id', 'is_repay']]
user_is_repay = user_is_repay.groupby('user_id').agg(lambda x:sum(x)/len(x)).reset_index()
user_tag = pd.merge(user_tag, user_is_repay, on='user_id', how='left')
user_tag['taglist'] = '|' + user_tag['taglist'] + '|'
# tag_ratio = {}
# for i, tag in enumerate(tag_count.keys()):
#     tag_ratio[tag] = user_tag[user_tag.taglist.str.contains('\|'+tag+'\|')]['is_repay'].sum()
#     tag_ratio[tag] = tag_ratio[tag] / tag_count[tag]
with open(path_data+file_tag_ratio, 'r') as f:
    tag_ratio = json.load(f)
for colname in ['_mean', '_max', '_min', '_50']:
    user_tag['tag_ratio'+colname] = user_tag.taglist.apply(get_stat_1, args=(tag_ratio, colname))    

# 用户标签-删除无用列
user_tag.drop(columns=['taglist', 'is_repay'], inplace=True)

# 用户借款-记录数
user_repay_count = df_user_repay_logs[df_user_repay_logs.user_id.isin(df_test.user_id.unique())][['user_id']]
user_repay_count['user_repay_count'] = 1
user_repay_count = user_repay_count.groupby('user_id').agg(sum).reset_index()

# 用户借款-频率
user_repay_freq = df_user_repay_logs[df_user_repay_logs.user_id.isin(df_test.user_id.unique())][['user_id', 'due_date']]
user_repay_freq['due_date'] = user_repay_freq['due_date'].apply(get_time_string)
user_repay_freq = user_repay_freq.groupby('user_id').agg(lambda x:':'.join(x)).reset_index()
user_repay_freq['due_date'] = user_repay_freq['due_date'].apply(get_max_time_gap)
user_repay_freq.replace({0:np.nan}, inplace=True)
user_repay_freq['due_date'] = user_repay_count['user_repay_count'] / user_repay_freq['due_date']
user_repay_freq.rename(columns={'due_date':'user_repay_freq'}, inplace=True)

# 用户借款-最大/最小/平均时间间隔
user_repay_interval = df_user_repay_logs[df_user_repay_logs.user_id.isin(df_test.user_id.unique())][['user_id', 'due_date']]
user_repay_interval['due_date'] = user_repay_interval['due_date'].apply(get_time_string)
user_repay_interval = user_repay_interval.groupby('user_id').agg(lambda x:':'.join(x)).reset_index()
user_repay_interval['user_repay_interval_max'] = user_repay_interval['due_date'].apply(get_repay_interval_1, args=('max', ))
user_repay_interval['user_repay_interval_min'] = user_repay_interval['due_date'].apply(get_repay_interval_1, args=('min', ))
user_repay_interval['user_repay_interval_avg'] = user_repay_interval['due_date'].apply(get_repay_interval_1, args=('avg', ))

# 用户借款-期数统计数据
user_repay_order = df_user_repay_logs[df_user_repay_logs.user_id.isin(df_test.user_id.unique())][['user_id', 'order_id']]
user_repay_order = get_stat(user_repay_order, 'user_id', 'order_id', 'user_repay_order')
user_repay_order.drop(columns='user_repay_order_mode', inplace=True)

# 用户借款-还款金额统计数据
user_repay_amt = df_user_repay_logs[df_user_repay_logs.user_id.isin(df_test.user_id.unique())][['user_id', 'due_amt']]
user_repay_amt = get_stat(user_repay_amt, 'user_id', 'due_amt', 'user_repay_amt')
user_repay_amt.drop(columns='user_repay_amt_mode', inplace=True)

# 用户借款-还款间隔统计数据
user_repay_gap = df_user_repay_logs[(df_user_repay_logs.user_id.isin(df_test.user_id.unique())) & (df_user_repay_logs.repay_date > 0)][['user_id', 'repay_date']]
user_repay_gap = get_stat(user_repay_gap, 'user_id', 'repay_date', 'user_repay_gap')
user_repay_gap.drop(columns='user_repay_gap_mode', inplace=True)

# 用户借款-用户还款率
user_repay_ratio = df_user_repay_logs[df_user_repay_logs.user_id.isin(df_test.user_id.unique())][['user_id', 'is_repay']]
user_repay_ratio = user_repay_ratio.groupby('user_id').agg(lambda x:sum(x)/len(x)).reset_index()
user_repay_ratio.rename(columns={'is_repay':'user_repay_ratio'}, inplace=True)

# 用户借款-合并
user_repay = df_user_repay_logs[df_user_repay_logs.user_id.isin(df_test.user_id.unique())][['user_id']].drop_duplicates()
user_repay = pd.merge(user_repay, user_repay_count, on='user_id', how='left')
user_repay = pd.merge(user_repay, user_repay_freq, on='user_id', how='left')
user_repay = pd.merge(user_repay, user_repay_interval, on='user_id', how='left')
user_repay = pd.merge(user_repay, user_repay_ratio, on='user_id', how='left')
user_repay = pd.merge(user_repay, user_repay_order, left_on='user_id', right_index=True, how='left')
user_repay = pd.merge(user_repay, user_repay_amt, left_on='user_id', right_index=True, how='left')
user_repay = pd.merge(user_repay, user_repay_gap, left_on='user_id', right_index=True, how='left')
user_repay.drop(columns=['due_date'], inplace=True)

## 标签维度

listing = pd.merge(df_test[['listing_id']], df_listing_info[['listing_id', 'term', 'rate', 'principal']], on='listing_id', how='left')
listing['listing_profit'] = listing['principal'] * listing['rate']
listing['listing_repay'] = listing['listing_profit'] / listing['term']

## 构造训练集特征
user = df_test[['user_id']].drop_duplicates()
user = pd.merge(user, user_gender, on='user_id', how='left')
user = pd.merge(user, user_age, on='user_id', how='left')
user = pd.merge(user, user_cell_province, on='user_id', how='left')
user = pd.merge(user, user_id_province, on='user_id', how='left')
user = pd.merge(user, user_basic, on='user_id', how='left')
user = pd.merge(user, user_beha, on='user_id', how='left')
user = pd.merge(user, user_tag, on='user_id', how='left')
user = pd.merge(user, user_repay, on='user_id', how='left')
df_test_feat = pd.merge(df_test, user, on='user_id', how='left')
df_test_feat = pd.merge(df_test_feat, listing, on='listing_id', how='left')
df_test_feat.drop(columns=['auditing_date'], inplace=True)
df_test_feat['age_18'] = 0
df_test_feat.to_csv(path_data+file_test_feat, index=None)