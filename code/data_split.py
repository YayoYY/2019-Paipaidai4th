'''Trainset and Valset is being constructed in this file'''

import warnings
from config import *
from functions import *
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

## reading
print('reading...')

# 训练集
df_train = pd.read_csv(path_data+file_train_feat)
df_train = pd.read_csv(path_data+file_train_feat)
df_train.repay_date.fillna(32, inplace=True)
df_train.repay_date = df_train.repay_date.astype(int)
# 划分
train, val = train_test_split(df_train, test_size=0.01)

## constructing
print('constructing...')

# 生成sub
train_raw = pd.read_csv(path_data+file_train)
sub = train_raw[train_raw.listing_id.isin(val.listing_id)]
sub.due_date = sub.due_date.astype('datetime64')
del sub['repay_date'], sub['repay_amt']
# 生成val_submission_sample
val_pred = sub.copy()
val_pred = val_pred.sort_values(by='listing_id')
val_pred.due_date = val_pred.due_date.apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
val_submission_sample = val_pred[['listing_id', 'auditing_date', 'due_date']]
val_submission_sample['repay_amt'] = 0
for i in range(sub.shape[0]):
    print(str(i)+' ', end='')
    listing_id = val_pred.iloc[i, :]['listing_id']
    auditing_date = datetime.strptime(val_pred.iloc[i, :]['auditing_date'], '%Y-%m-%d')
    due_date = datetime.strptime(val_pred.iloc[i, :]['due_date'], '%Y-%m-%d')
    for day in range(1, (due_date-auditing_date).days):
        val_submission_sample = val_submission_sample.append(pd.Series({'listing_id': listing_id, 
                                   'auditing_date': datetime.strftime(auditing_date, '%Y-%m-%d'), 
                                   'due_date': datetime.strftime(auditing_date+timedelta(days=day), '%Y-%m-%d')}), ignore_index=True)
val_submission_sample = val_submission_sample.sort_values(by=['listing_id', 'auditing_date', 'due_date'])
val_submission_sample.drop_duplicates(inplace=True)
val_submission_sample.rename(columns={'due_date': 'repay_date'}, inplace=True)
del val_submission_sample['auditing_date']

# 备份
train.to_csv(path_data+file_trainset, index=None)
val.to_csv(path_data+file_valset, index=None)
val_submission_sample.to_csv(path_data+file_val_submission_sample, index=None)