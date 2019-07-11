'''LightGBM & XGBoost model is being trained here'''

import warnings
from config import *
from functions import *
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from lightgbm.sklearn import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

# reading
print('reading...')

clf = joblib.load('treemodel/lgb_195feat.model')
booster = clf.booster_
importances = booster.feature_importance(importance_type='split')
feature_names = booster.feature_name()
dic = {x:y for (x,y) in zip(feature_names, importances)}
dic = sorted(dic.items(), key=lambda dic:dic[1], reverse=True)
feature_cols = [item[0] for item in dic][:40]
train = pd.read_csv(path_data+file_trainset)
val = pd.read_csv(path_data+file_valset)
test = pd.read_csv(path_data+file_test_feat)
del_cols = ['user_repay_amt_25', 'user_repay_amt_50', 'user_repay_amt_75',
            'user_repay_amt_min', 'user_repay_amt_mean' ,'tag_ratio_mean', 'tag_ratio_50', 'user_repay_gap_75']
del_cols += ['id_province_repay_ratio', 'cell_province_repay_gap_mean','user_repay_freq',
            'user_repay_interval_max']
del_cols += ['listing_repay']
for col in del_cols:
    feature_cols.remove(col)


# adding features
print('adding features...')

user_info = pd.read_csv(path_data+file_user_info_last_date)
le = LabelEncoder()
user_info.id_province = le.fit_transform(user_info.id_province)
le = LabelEncoder()
user_info.gender = le.fit_transform(user_info.gender)
le = LabelEncoder()
user_info.id_city = le.fit_transform(user_info.id_city)

user_repay = pd.read_csv(path_data+file_user_repay_logs)
user_repay_max_order = user_repay.groupby('user_id', as_index=False)['order_id'].max()
user_repay_avg_order = user_repay.groupby('user_id', as_index=False)['order_id'].agg(lambda x:sum(x)/len(x))
user_repay_unique_listing = user_repay.groupby('user_id')['listing_id'].nunique().reset_index()
user_repay_is_repay = user_repay.groupby('user_id', as_index=False)['repay_date'].count()
user_repay_num = user_repay.groupby('user_id')['repay_date'].size().reset_index()
user_repay_is_repay['user_repay_ratio_2'] = user_repay_is_repay['repay_date'] / user_repay_num['repay_date']
user_repay_max_due_amt = user_repay.groupby('user_id', as_index=False)['due_amt'].max()
user_repay_sum_due_amt = user_repay.groupby('user_id', as_index=False)['due_amt'].sum()
user_repay_max_order = rename(user_repay_max_order, 'order_id', 'user_repay_max_order')
user_repay_avg_order = rename(user_repay_avg_order, 'order_id', 'user_repay_avg_order')
user_repay_unique_listing = rename(user_repay_unique_listing, 'listing_id', 'user_repay_unique_listing')
user_repay_is_repay = rename(user_repay_is_repay, 'repay_date', 'user_repay_is_repay')
user_repay_num = rename(user_repay_num, 'repay_date', 'user_repay_num')
user_repay_max_due_amt = rename(user_repay_max_due_amt, 'due_amt', 'user_repay_max_due_amt')
user_repay_sum_due_amt = rename(user_repay_sum_due_amt, 'due_amt', 'user_repay_sum_due_amt')

train = train.merge(user_info[['user_id', 'id_province', 'id_city', 'gender']], on='user_id', how='left')
val = val.merge(user_info[['user_id', 'id_province', 'id_city', 'gender']], on='user_id', how='left')
test = test.merge(user_info[['user_id', 'id_province', 'id_city', 'gender']], on='user_id', how='left')
train = train.merge(user_repay_max_order, on='user_id', how='left')
train = train.merge(user_repay_avg_order, on='user_id', how='left')
train = train.merge(user_repay_unique_listing, on='user_id', how='left')
train = train.merge(user_repay_is_repay, on='user_id', how='left')
train = train.merge(user_repay_num, on='user_id', how='left')
train = train.merge(user_repay_max_due_amt, on='user_id', how='left')
train = train.merge(user_repay_sum_due_amt, on='user_id', how='left')
val = val.merge(user_repay_max_order, on='user_id', how='left')
val = val.merge(user_repay_avg_order, on='user_id', how='left')
val = val.merge(user_repay_unique_listing, on='user_id', how='left')
val = val.merge(user_repay_is_repay, on='user_id', how='left')
val = val.merge(user_repay_num, on='user_id', how='left')
val = val.merge(user_repay_max_due_amt, on='user_id', how='left')
val = val.merge(user_repay_sum_due_amt, on='user_id', how='left')
test = test.merge(user_repay_max_order, on='user_id', how='left')
test = test.merge(user_repay_avg_order, on='user_id', how='left')
test = test.merge(user_repay_unique_listing, on='user_id', how='left')
test = test.merge(user_repay_is_repay, on='user_id', how='left')
test = test.merge(user_repay_num, on='user_id', how='left')
test = test.merge(user_repay_max_due_amt, on='user_id', how='left')
test = test.merge(user_repay_sum_due_amt, on='user_id', how='left')
    
for col in ['id_province', 'id_city', 'gender']:
    feature_names.append(col)
for col in ['user_repay_max_order', 'user_repay_avg_order', 'user_repay_unique_listing', 'user_repay_is_repay', 
        'user_repay_num', 'user_repay_ratio_2', 'user_repay_max_due_amt', 'user_repay_sum_due_amt']:
    feature_names.append(col)

train_y = train.repay_date.values
train_X = train[feature_names]
val_y = val.repay_date.values
val_X = val[feature_names]
test_X = test[feature_names]

# training
print('training...')

clf = LGBMClassifier(
    learning_rate = 0.2,
    n_estimators = 1000,
    subsample = 0.4,
    subsample_freq = 1,
    colsample_bytree = 0.4,
    random_state = 2019,
    num_leaves = 10,
    min_child_samples = 20,
    max_depth = 3
)
clf.fit(train_X, train_y, \
        eval_set=[(train_X, train_y), (val_X, val_y)], \
        early_stopping_rounds=10)
joblib.dump(clf, 'treemodel/lgb_final.model')


clf_xgb = XGBClassifier(
    learning_rate = 0.2,
    n_estimators = 1000,
    subsample = 0.4,
    subsample_freq = 1,
    colsample_bytree = 0.4,
    random_state = 2019,
    num_leaves = 10,
    min_child_samples = 20,
    max_depth = 3
)
clf_xgb.fit(train_X, train_y, \
        eval_set=[(train_X, train_y), (val_X, val_y)], \
        early_stopping_rounds=10)
joblib.dump(clf, 'treemodel/xgb.model')

# predict
print('predict...')
test_pred_prob_1 = clf.predict_proba(test_X, num_iteration=clf.best_iteration_)
test_pred_prob_2 = clf_xgb.predict_proba(test_X)
test_pred_prob = (test_pred_prob_1 + test_pred_prob_2) / 2
sub = pd.read_csv(path_data+file_test, parse_dates=['due_date'])
prob_cols = ['prob_{}'.format(i) for i in range(33)]
for i, f in enumerate(prob_cols):
    sub[f] = test_pred_prob[:, i]
sub_example = pd.read_csv('../result/submission_sample.csv', parse_dates=['repay_date'])
sub_example = pd.merge(sub_example, sub, how='left', on='listing_id')
sub_example['days'] = (sub_example['due_date'] - sub_example['repay_date']).dt.days
test_prob = sub_example[prob_cols].values
test_labels = sub_example['days'].values
test_prob = [test_prob[i][test_labels[i]] for i in range(test_prob.shape[0])]
sub_example['repay_amt'] = sub_example['due_amt'] * test_prob
sub_example[['listing_id', 'repay_date', 'repay_amt']].to_csv('sub.csv', index=False)