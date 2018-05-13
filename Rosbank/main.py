import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from datetime import datetime
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from tools import write_submission, preprocessing
from imblearn.over_sampling import ADASYN
from currency_converter import CurrencyConverter
from datetime import date

# load data
path_to_data = 'data/'
train = pd.read_csv(os.path.join(path_to_data,'train.csv'))
X_test = pd.read_csv(os.path.join(path_to_data,'test.csv'))
test_id = X_test.cl_id

target = train.groupby(['cl_id'])[['target_flag', 'target_sum']].max()

#preprocessing
X_train = preprocessing(train)
X_test = preprocessing(X_test)

# train = csr_matrix(hstack([data['amount'].values.reshape(-1,1), data_MCC, data_chanel_type, data_currency, data_trx_category]))[:target.shape[0]]
# X_test = csr_matrix(hstack([data['amount'].values.reshape(-1,1), data_MCC, data_chanel_type, data_currency, data_trx_category]))[target.shape[0]:]
#
X_train, X_val, y_train, y_val = train_test_split(X_train, target.target_flag, test_size=0.2, random_state=17, shuffle=True)

# adasyn = ADASYN(random_state=17)
# X_train,y_train = adasyn.fit_sample(X_train,y_train)
# stf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
# print (cross_val_score(LogisticRegression(penalty='l2'),X_train,y_train, cv=stf, scoring='roc_auc'))

import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

params = {
    #'max_depth': 8,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'is_unbalance' : True,
    'metric': {'l2', 'auc'},
    'lambda_l2' : 1,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=300,
                valid_sets=lgb_eval,
                early_stopping_rounds=21)

# print('Save model...')
# # save model to file
# gbm.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
write_submission(y_pred, test_id, 'sub/lgb.csv')


