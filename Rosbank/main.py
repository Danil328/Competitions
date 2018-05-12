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

from tools import write_submission
from imblearn.over_sampling import ADASYN
from currency_converter import CurrencyConverter
from datetime import date

# load data
path_to_data = 'data/'
train = pd.read_csv(os.path.join(path_to_data,'train.csv'))
X_test = pd.read_csv(os.path.join(path_to_data,'test.csv'))

target = train[['target_flag', 'target_sum']]
X_train = train.drop(columns = ['target_flag', 'target_sum'], axis = 1)


#preprocessing
def preprocessing(df):
    df.index = range(len(df))
    # date preprocess
    #df['year'] = df['PERIOD'].map(lambda x: datetime.strptime(x, "%d/%m/%Y").year)
    df['month'] = df['PERIOD'].map(lambda x: datetime.strptime(x, "%d/%m/%Y").month)
    #df['day'] = df['PERIOD'].map(lambda x: datetime.strptime(x, "%d/%m/%Y").day)
    #df['day_of_week'] = df['PERIOD'].map(lambda x: datetime.strptime(x, "%d/%m/%Y").weekday())

    df['TR_year'] = df['TRDATETIME'].map(lambda x: datetime.strptime(x, "%d%b%y:%H:%M:%S").year)
    df['TR_month'] = df['TRDATETIME'].map(lambda x: datetime.strptime(x, "%d%b%y:%H:%M:%S").month)
    df['TR_day'] = df['TRDATETIME'].map(lambda x: datetime.strptime(x, "%d%b%y:%H:%M:%S").day)
    df['TR_day_of_week'] = df['TRDATETIME'].map(lambda x: datetime.strptime(x, "%d%b%y:%H:%M:%S").weekday())
    df['TR_hour'] = df['TRDATETIME'].map(lambda x: datetime.strptime(x, "%d%b%y:%H:%M:%S").hour)

    df.drop(columns=['PERIOD','TRDATETIME', 'cl_id'], axis=1, inplace=True)

    c = CurrencyConverter(os.path.join(path_to_data,'eurofxref-hist.csv'))
    codes = pd.read_csv(os.path.join(path_to_data,'codes.csv'))
    codes.fillna(0,inplace=True)
    codes.currency_code = codes.currency_code.apply(int)
    amount = []
    for i in tqdm(range(len(df['amount']))):
        amount.append(c.convert(df['amount'][i], codes[codes['currency_code']==df['currency'][i]]['currency_ABC'].values[0], 'USD'))
    df['amount'] = amount

    date_columns = ['month','TR_year','TR_month','TR_day','TR_day_of_week','TR_hour']
    for i in date_columns:
        df[i] = StandardScaler().fit_transform(df[i].reshape(-1,1))

    df.amount = MinMaxScaler().fit_transform(df.amount.values.reshape(-1,1))
    df['balance'] = df['trx_category'].map({'POS': -1, 'C2C_OUT': -1, 'C2C_IN': 1, 'DEPOSIT': 1, 'WD_ATM_PARTNER': -1,
                                            'WD_ATM_ROS': -1, 'WD_ATM_OTHER': -1, 'BACK_TRX': 1,'CAT': 1, 'CASH_ADV': -1}) * df.amount
    df = df.fillna('type')

    return df

data = preprocessing(pd.concat([X_train,X_test]))

le = LabelEncoder()
oe = OneHotEncoder()
data['MCC'] = le.fit_transform(data['MCC'])
data_MCC = oe.fit_transform(data['MCC'].values.reshape(-1, 1))

data['channel_type'] = le.fit_transform(data['channel_type'])
data_chanel_type = oe.fit_transform(data['channel_type'].values.reshape(-1,1))

data['currency'] = le.fit_transform(data['currency'])
data_currency = oe.fit_transform(data['currency'].values.reshape(-1,1))

data['trx_category'] = le.fit_transform(data['trx_category'])
data_trx_category = oe.fit_transform(data['trx_category'].values.reshape(-1,1))

train = csr_matrix(hstack([data['amount'].values.reshape(-1,1), data_MCC, data_chanel_type, data_currency, data_trx_category]))[:target.shape[0]]
X_test = csr_matrix(hstack([data['amount'].values.reshape(-1,1), data_MCC, data_chanel_type, data_currency, data_trx_category]))[target.shape[0]:]

X_train, X_val, y_train, y_val = train_test_split(train, target.target_flag, test_size=0.2, random_state=17, shuffle=False)
stf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

# adasyn = ADASYN(random_state=17)
# X_train,y_train = adasyn.fit_sample(X_train,y_train)

print (cross_val_score(LogisticRegression(penalty='l2'),X_train,y_train, cv=stf, scoring='roc_auc'))

lr = LogisticRegression(penalty='l2')
lr.fit(X_train, y_train)
pred = lr.predict_proba(X_val)[:,-1]
print (roc_auc_score(y_val, pred))
# write_submission(pred, 'sub/1.csv')
