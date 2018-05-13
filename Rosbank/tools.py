import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from datetime import datetime
from tqdm import tqdm


from currency_converter import CurrencyConverter
from datetime import date

path_to_data = 'data/'

def preprocessing(df):
    # date preprocess
    df['year'] = df['PERIOD'].map(lambda x: datetime.strptime(x, "%d/%m/%Y").year)
    df['month'] = df['PERIOD'].map(lambda x: datetime.strptime(x, "%d/%m/%Y").month)
    #df['day'] = df['PERIOD'].map(lambda x: datetime.strptime(x, "%d/%m/%Y").day)
    #df['day_of_week'] = df['PERIOD'].map(lambda x: datetime.strptime(x, "%d/%m/%Y").weekday())

    df['TRDATETIME'] = df['TRDATETIME'].map(lambda x: datetime.strptime(x, "%d%b%y:%H:%M:%S"))
    df['TR_year'] = df['TRDATETIME'].map(lambda x: x.year)
    df['TR_month'] = df['TRDATETIME'].map(lambda x: x.month)
    df['TR_day'] = df['TRDATETIME'].map(lambda x: x.day)
    df['TR_day_of_week'] = df['TRDATETIME'].map(lambda x: x.weekday())
    df['TR_weekend'] = df['TR_day_of_week'].apply(lambda x: 1 if x == 5 or x==6 else 0)
    df['TR_hour'] = df['TRDATETIME'].map(lambda x: x.hour)

    df.drop(columns=['PERIOD'], axis=1, inplace=True)

    c = CurrencyConverter(os.path.join(path_to_data,'eurofxref-hist.csv'))
    codes = pd.read_csv(os.path.join(path_to_data,'codes.csv'))
    codes.fillna(0,inplace=True)
    codes.currency_code = codes.currency_code.apply(int)
    amount = []
    for i in tqdm(range(len(df['amount']))):
        amount.append(c.convert(df['amount'][i], codes[codes['currency_code']==df['currency'][i]]['currency_ABC'].values[0], 'USD'))
    df['amount'] = amount

    # date_columns = ['month','TR_year','TR_month','TR_day','TR_day_of_week','TR_hour']
    # for i in date_columns:
    #     df[i] = StandardScaler().fit_transform(df[i].reshape(-1,1))

    #df.amount = MinMaxScaler().fit_transform(df.amount.values.reshape(-1,1))
    df['balance'] = df['trx_category'].map({'POS': -1, 'C2C_OUT': -1, 'C2C_IN': 1, 'DEPOSIT': 1, 'WD_ATM_PARTNER': -1,
                                            'WD_ATM_ROS': -1, 'WD_ATM_OTHER': -1, 'BACK_TRX': 1,'CAT': -1, 'CASH_ADV': -1}) * df.amount
    df = df.fillna('type')

    new_data = pd.DataFrame()
    new_data['cl_id'] = df.groupby(['cl_id'])['cl_id'].max()
    new_data['mean_amount'] = df.groupby(['cl_id'])['amount'].mean()
    new_data['max_amount'] = df.groupby(['cl_id'])['amount'].max()
    new_data['min_amount'] = df.groupby(['cl_id'])['amount'].min()
    new_data['delta_amount'] = new_data['max_amount'] - new_data['min_amount']
    new_data['balance'] = df.groupby(['cl_id'])['balance'].mean()
    new_data['delta_first_end_TR'] = (df.groupby(['cl_id'])['TRDATETIME'].max()-df.groupby(['cl_id'])['TRDATETIME'].min()).apply(lambda x: x.days)
    new_data['mode_channel_type'] = df.groupby(['cl_id'])['channel_type'].max()
    new_data['mode_trx_category'] = df.groupby(['cl_id'])['trx_category'].max()
    new_data['mode_currency'] = df.groupby(['cl_id'])['currency'].max()

    le = LabelEncoder()
    oe = OneHotEncoder()

    new_data['mode_channel_type'] = le.fit_transform(new_data['mode_channel_type'])
    mode_channel_type = oe.fit_transform(new_data['mode_channel_type'].values.reshape(-1,1)).todense()
    mode_channel_type = pd.DataFrame(mode_channel_type, columns=list(le.classes_))
    mode_channel_type['cl_id'] = df.cl_id
    mode_channel_type = mode_channel_type.groupby(['cl_id']).sum()

    new_data = pd.concat([new_data,mode_channel_type], axis=1)

    new_data['mode_trx_category'] = le.fit_transform(new_data['mode_trx_category'])
    mode_trx_category = oe.fit_transform(new_data['mode_trx_category'].values.reshape(-1,1)).todense()
    mode_trx_category = pd.DataFrame(mode_trx_category, columns=list(le.classes_))
    mode_trx_category['cl_id'] = df.cl_id
    mode_trx_category = mode_trx_category.groupby(['cl_id']).sum()

    new_data = pd.concat([new_data,mode_trx_category], axis=1)

    new_data['mode_currency'] = le.fit_transform(new_data['mode_currency'])
    mode_currency = oe.fit_transform(new_data['mode_currency'].values.reshape(-1,1)).todense()
    mode_currency = pd.DataFrame(mode_currency, columns=list(le.classes_))
    mode_currency['cl_id'] = df.cl_id
    mode_currency = mode_currency.groupby(['cl_id']).sum()

    new_data = pd.concat([new_data,mode_currency], axis=1)

    new_data.drop(columns = ['mode_channel_type','mode_trx_category','mode_currency'], axis=1, inplace=True)
    # df['MCC'] = le.fit_transform(df['MCC'])
    # data_MCC = oe.fit_transform(df['MCC'].values.reshape(-1, 1))

    df['channel_type'] = le.fit_transform(df['channel_type'])
    data_chanel_type = oe.fit_transform(df['channel_type'].values.reshape(-1,1)).todense()
    data_chanel_type = pd.DataFrame(data_chanel_type, columns=list(le.classes_))
    data_chanel_type['cl_id'] = df.cl_id
    data_chanel_type = data_chanel_type.groupby(['cl_id']).sum()

    new_data = pd.concat([new_data,data_chanel_type], axis=1)

    df['currency'] = le.fit_transform(df['currency'])
    data_currency = oe.fit_transform(df['currency'].values.reshape(-1,1)).todense()
    data_currency = pd.DataFrame(data_currency, columns=list(le.classes_))
    data_currency['cl_id'] = df.cl_id
    data_currency = data_currency.groupby(['cl_id']).sum()

    new_data = pd.concat([new_data,data_currency], axis=1)


    df['trx_category'] = le.fit_transform(df['trx_category'])
    data_trx_category = oe.fit_transform(df['trx_category'].values.reshape(-1,1)).todense()
    data_trx_category = pd.DataFrame(data_trx_category, columns=list(le.classes_))
    data_trx_category['cl_id'] = df.cl_id
    data_trx_category = data_trx_category.groupby(['cl_id']).sum()

    new_data = pd.concat([new_data,data_trx_category], axis=1)

    new_data.drop(columns = ['cl_id'], axis=1, inplace=True)

    return new_data

def write_submission(pred, test_id, path):
    df = pd.DataFrame(columns = ['_ID_','_VAL_'])
    df['_ID_'] = test_id
    df['_VAL_'] = pred
    df.to_csv(path, index=False)
