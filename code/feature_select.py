'''Using LightGBM to rank the importances of features'''

import warnings
from config import *
from functions import *
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from lightgbm.sklearn import LGBMClassifier
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')

## reading
print('reading...')


train = pd.read_csv(path_data+file_trainset)
val = pd.read_csv(path_data+file_valset)
    
train_y = train.repay_date.values
train_X = train.drop(columns=['repay_date', 'user_id', 'listing_id'])
val_y = val.repay_date.values
val_X = val.drop(columns=['repay_date', 'user_id', 'listing_id'])

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

# saving
print('saving...')

joblib.dump(clf_xgb, 'treemodel/lgb_195feat.model')