'''Null value is being filled in this file'''

import warnings
from config import *
from functions import *
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMClassifier
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')

## reading
print('reading...')

booster = joblib.load('treemodel/lgb_195feat.model').booster_
importances = booster.feature_importance(importance_type='split')
feature_names = booster.feature_name()
dic = {x:y for (x,y) in zip(feature_names, importances)}
dic = sorted(dic.items(), key=lambda dic:dic[1], reverse=True)
feature_cols = [item[0] for item in dic][:50]
feature_cols.remove('least_100')

train = pd.read_csv(path_data+file_trainset)
val = pd.read_csv(path_data+file_valset)
train_y = train.repay_date.values
train_X = train[feature_cols]
val_y = val.repay_date.values
val_X = val[feature_cols]
test_X = pd.read_csv(path_data+file_test_feat)[feature_cols]

## cleaning
print('cleaning...')

# 删除一些列
del_columns = ['tag_ratio_max', 'tag_ratio_50', 'tag_ratio_mean', ]
train_X.drop(columns=del_columns, inplace=True)
val_X.drop(columns=del_columns, inplace=True)
test_X.drop(columns=del_columns, inplace=True)

train_X = fill0(train_X)
test_X = fill0(test_X)
val_X = fill0(val_X)

isnull_columns = train_X.columns[train_X.isnull().sum() > 0 ]
notnull_columns = train_X.columns[train_X.isnull().sum() == 0]

# 训练rf模型
for column in isnull_columns:
    print('column\t', column)
    data = pd.concat([train_X, val_X, test_X], axis=0)
    tmpX = data[data[column].notnull()][notnull_columns]
    tmpy = data[data[column].notnull()][column]
    reg = RandomForestRegressor(max_features=len(notnull_columns), max_depth=None, min_samples_split=2, n_jobs=-1)
    reg.fit(tmpX, tmpy)
    joblib.dump(reg, 'rf/'+column+'.model')

train_X = fillrf(train_X, isnull_columns, notnull_columns)
val_X = fillrf(val_X, isnull_columns, notnull_columns)
test_X = fillrf(test_X, isnull_columns, notnull_columns)

train_X.to_scv(path_data+file_train_X_fillna, index=None)
val_X.to_scv(path_data+file_vaj_X_fillna, index=None)
test_X.to_scv(path_data+file_test_X_fillna, index=None)