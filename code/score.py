'''scoring'''

import warnings
from config import *
from functions import *
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

## defining
print('defining...')

clf = joblib.load('treemodel/lgb_final.model')
clf_xgb = joblib.load('treemodel/xgb.model')

def rmse(y_true, y_pred):
    cols = ['listing_id', 'repay_date', 'repay_amt']
    y_true['max_gap'] = y_true['auditing_date'].astype(str) + ':' + y_true['due_date'].astype(str)
    y_true['max_gap'] = y_true['max_gap'].apply(get_auditing_repay_gap)
    max_gap = y_true[['listing_id', 'max_gap']].drop_duplicates()
    y_pred.repay_date = y_pred.repay_date.astype(str)
    se = pd.merge(y_true[cols], y_pred[cols], on=['listing_id', 'repay_date'], how='outer').fillna(0)
    se.replace({'\\N':np.nan}, inplace=True)
    se.repay_amt_x = se.repay_amt_x.astype(float)
    se['repay_amt_diff'] = np.power((se['repay_amt_x'] - se['repay_amt_y']), 2)
    se = se.groupby('listing_id', as_index=False)['repay_amt_diff'].sum()
    df = max_gap.merge(se, on='listing_id', how='outer')
    df['rmse'] = df['repay_amt_diff'] / df['max_gap']
    return df, df.rmse.sum() / df.listing_id.count()

## reading
print('reading...')

# 特征
booster = clf.booster_
feature_cols = booster.feature_name()

# 训练集
train_raw = pd.read_csv(path_data+file_train, parse_dates=['due_date', 'repay_date'])
# 验证特征
val = pd.read_csv(path_data+file_valset)
val_X = val[feature_cols]
# 提交格式验证集
val_submission_sample = pd.read_csv(path_data+file_val_submission_sample, parse_dates=['repay_date'])

## predicting
print('predicting...')

test_pred_prob_1 = clf.predict_proba(val_X, num_iteration=clf.best_iteration_)
test_pred_prob_2 = clf_xgb.predict_proba(val_X)
test_pred_prob = (test_pred_prob_1 + test_pred_prob_2) / 2
sub = train_raw[train_raw.listing_id.isin(val.listing_id)]
del sub['repay_date'], sub['repay_amt']
prob_cols = ['prob_{}'.format(i) for i in range(33)]
for i, f in enumerate(prob_cols):
    sub[f] = test_pred_prob[:, i]
y_val_pred = val_submission_sample.copy()
y_val_pred = pd.merge(y_val_pred, sub, how='left', on='listing_id')
y_val_pred['days'] = (y_val_pred['due_date'] - y_val_pred['repay_date']).dt.days
test_prob = y_val_pred[prob_cols].values
test_labels = y_val_pred['days'].values
test_prob = [test_prob[i][test_labels[i]] for i in range(test_prob.shape[0])]
y_val_pred['repay_amt'] = y_val_pred['due_amt'] * test_prob

## estimating
print('estimating...')

val_raw = train_raw[train_raw.listing_id.isin(val.listing_id)]
true = val_raw[['listing_id', 'auditing_date','due_date','repay_date', 'repay_amt']]
pred = y_val_pred[['listing_id', 'auditing_date','due_date','repay_date', 'repay_amt']]
df, score = rmse(true, pred)

print(score)