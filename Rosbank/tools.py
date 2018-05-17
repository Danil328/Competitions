import requests
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler, RobustScaler
from datetime import datetime
from tqdm import tqdm


from currency_converter import CurrencyConverter
from datetime import date

path_to_data = 'data/'

def preprocessing(df):
    # date preprocess
    df.index = df.reindex()
    df['year'] = df['PERIOD'].map(lambda x: datetime.strptime(x, "%d/%m/%Y").year)
    df['month'] = df['PERIOD'].map(lambda x: datetime.strptime(x, "%d/%m/%Y").month)
    df['TRDATETIME'] = df['TRDATETIME'].map(lambda x: datetime.strptime(x, "%d%b%y:%H:%M:%S"))
    df['TR_year'] = df['TRDATETIME'].map(lambda x: x.year)
    df['TR_month'] = df['TRDATETIME'].map(lambda x: x.month)
    df['TR_day'] = df['TRDATETIME'].map(lambda x: x.day)
    df['TR_day_of_week'] = df['TRDATETIME'].map(lambda x: x.weekday())
    df['TR_weekend'] = df['TR_day_of_week'].apply(lambda x: 1 if x == 5 or x==6 else 0)
    df['TR_hour'] = df['TRDATETIME'].map(lambda x: x.hour)

    df.drop(columns=['PERIOD'], axis=1, inplace=True)

    mcc_codes = pd.read_csv(path_to_data+'/mcch.csv')
    mcc_codes.index = mcc_codes.MCC
    df['mcc_group'] = df['MCC'].apply(lambda x: mcc_codes.group[x])

    # c = CurrencyConverter(os.path.join(path_to_data,'eurofxref-hist.csv'))
    # codes = pd.read_csv(os.path.join(path_to_data,'codes.csv'))
    # codes.fillna(0,inplace=True)
    # codes.currency_code = codes.currency_code.apply(int)
    # for i in tqdm(range(len(df['amount']))):
    #     if df['currency'][i]!=810:
    #         df['amount'][i] = c.convert(df['amount'][i], codes[codes['currency_code']==df['currency'][i]]['currency_ABC'].values[0], 'RUB')

    df['balance'] = df['trx_category'].map({'POS': -1, 'C2C_OUT': -1, 'C2C_IN': 1, 'DEPOSIT': 1, 'WD_ATM_PARTNER': -1,
                                            'WD_ATM_ROS': -1, 'WD_ATM_OTHER': -1, 'BACK_TRX': 1,'CAT': -1, 'CASH_ADV': -1}) * df.amount
    df = df.fillna('type')

    new_data = pd.DataFrame()
    new_data['cl_id'] = df.groupby(['cl_id'])['cl_id'].max()
    new_data['count_TR'] = df.groupby(['cl_id'])['cl_id'].count()
    new_data['sum_TR'] = df.groupby(['cl_id'])['amount'].sum()
    new_data['mean_amount'] = df.groupby(['cl_id'])['amount'].mean()
    new_data['max_amount'] = df.groupby(['cl_id'])['amount'].max()
    new_data['min_amount'] = df.groupby(['cl_id'])['amount'].min()
    new_data['std_amount'] = df.groupby(['cl_id'])['amount'].std()
    new_data['delta_amount'] = new_data['max_amount'] - new_data['min_amount']
    new_data['mean_balance'] = df.groupby(['cl_id'])['balance'].mean()
    # new_data['balance'] = df.groupby(['cl_id'])['balance'].sum()
    new_data['sign'] = new_data['mean_balance'].apply(lambda x: 1 if x>0 else 0)
    new_data['delta_first_end_TR'] = (df.groupby(['cl_id'])['TRDATETIME'].max()-df.groupby(['cl_id'])['TRDATETIME'].min()).apply(lambda x: x.days)
    new_data['mode_channel_type'] = df.groupby(['cl_id'])['channel_type'].max()
    new_data['mode_trx_category'] = df.groupby(['cl_id'])['trx_category'].max()
    new_data['mode_currency'] = df.groupby(['cl_id'])['currency'].max()
    new_data['mode_mcc_group'] = df.groupby(['cl_id'])['mcc_group'].max()

    le = LabelEncoder()
    oe = OneHotEncoder()

    new_data['mode_channel_type'] = le.fit_transform(new_data['mode_channel_type'])
    mode_channel_type = oe.fit_transform(new_data['mode_channel_type'].values.reshape(-1,1)).todense()
    mode_channel_type = pd.DataFrame(mode_channel_type, columns=list(le.classes_))
    new_data = pd.concat([new_data,mode_channel_type], axis=1)
    #
    # new_data['mode_trx_category'] = le.fit_transform(new_data['mode_trx_category'])
    # mode_trx_category = oe.fit_transform(new_data['mode_trx_category'].values.reshape(-1,1)).todense()
    # mode_trx_category = pd.DataFrame(mode_trx_category, columns=list(le.classes_))
    # new_data = pd.concat([new_data,mode_trx_category], axis=1)

    # new_data['mode_currency'] = le.fit_transform(new_data['mode_currency'])
    # mode_currency = oe.fit_transform(new_data['mode_currency'].values.reshape(-1,1)).todense()
    # mode_currency = pd.DataFrame(mode_currency, columns=list(le.classes_))
    # new_data = pd.concat([new_data,mode_currency], axis=1)

    # new_data['mode_mcc_group'] = le.fit_transform(new_data['mode_mcc_group'])
    # mode_mcc_group = oe.fit_transform(new_data['mode_mcc_group'].values.reshape(-1,1)).todense()
    # mode_mcc_group = pd.DataFrame(mode_mcc_group, columns=list(le.classes_))
    # new_data = pd.concat([new_data,mode_mcc_group], axis=1)

    new_data.drop(columns = ['mode_channel_type','mode_trx_category','mode_currency','mean_balance','max_amount','min_amount', 'mode_mcc_group'], axis=1, inplace=True)

    # df['channel_type'] = le.fit_transform(df['channel_type'])
    # data_chanel_type = oe.fit_transform(df['channel_type'].values.reshape(-1,1)).todense()
    # data_chanel_type = pd.DataFrame(data_chanel_type, columns=list(le.classes_))
    # data_chanel_type['cl_id'] = df.cl_id.values
    # data_chanel_type = data_chanel_type.groupby(['cl_id']).mean()
    # new_data = pd.concat([new_data,data_chanel_type], axis=1)
    #
    # df['currency'] = le.fit_transform(df['currency'])
    # data_currency = oe.fit_transform(df['currency'].values.reshape(-1,1)).todense()
    # data_currency = pd.DataFrame(data_currency, columns=list(le.classes_))
    # data_currency['cl_id'] = df.cl_id.values
    # data_currency = data_currency.groupby(['cl_id']).sum()
    # new_data = pd.concat([new_data,data_currency], axis=1)
    #
    df['trx_category'] = le.fit_transform(df['trx_category'])
    data_trx_category = oe.fit_transform(df['trx_category'].values.reshape(-1,1)).todense()
    data_trx_category = pd.DataFrame(data_trx_category, columns=list(le.classes_))
    data_trx_category['cl_id'] = df.cl_id.values
    data_trx_category = data_trx_category.groupby(['cl_id']).sum()
    new_data = pd.concat([new_data, data_trx_category], axis=1)

    # df['mcc_group'] = le.fit_transform(df['mcc_group'])
    # data_mcc_group = oe.fit_transform(df['mcc_group'].values.reshape(-1,1)).todense()
    # data_mcc_group = pd.DataFrame(data_mcc_group, columns=list(le.classes_))
    # data_mcc_group = data_mcc_group.groupby(['cl_id']).sum()
    # new_data = pd.concat([new_data, data_mcc_group], axis=1)
    #
    # data_mcc_group = df.groupby('cl_id').apply(lambda x: x.groupby('mcc_group')['balance'].sum()).unstack().fillna(0)
    # data_mcc_group.rename(columns=lambda x: 'data_mcc_group_' + str(x), inplace=True)
    # new_data = pd.concat([new_data, data_mcc_group], axis=1)

    # "Базовая фича" - Количество покупок по каждой категории
    X = df.groupby('cl_id') \
        .apply(lambda x: x[['trx_category']].unstack().value_counts()) \
        .unstack() \
        .fillna(0)
    X.rename(columns=lambda x: 'trx_category_value_counts_' + str(x), inplace=True)
    new_data = pd.concat([new_data, X], axis=1)

    # Сумма покупок по каждой категории
    F2 = df.groupby('cl_id').apply(lambda x: x.groupby('trx_category')['amount'].sum()).unstack().fillna(0)
    F2.rename(columns=lambda x: 'sum_' + str(x), inplace=True)
    new_data = pd.concat([new_data, F2], axis=1)

    # Кол-во покупок по каждой категории
    F3 = df.groupby('cl_id').apply(lambda x: x.groupby('trx_category')['amount'].count()).unstack().fillna(0)
    F3.rename(columns=lambda x: 'count_' + str(x), inplace=True)
    new_data = pd.concat([new_data, F3], axis=1)

    # Средняя сумма покупок по дням недели
    F12 = df.groupby('cl_id').apply(lambda x: x.groupby('TR_day_of_week')['amount'].mean()).unstack().fillna(0)
    F12.rename(columns=lambda x: 'mean_df_count_by_week_day_' + str(x), inplace=True)
    new_data = pd.concat([new_data, F12], axis=1)

    # Транзакции по часам
    # F13 = df.groupby('cl_id').apply(lambda x: x.groupby('TR_hour')['amount'].mean()).unstack().fillna(0)
    # F13.rename(columns=lambda x: 'mean_df_count_by_h_' + str(x), inplace=True)
    # new_data = pd.concat([new_data, F13], axis=1)

    return new_data

def preprocessing_sber(df):
    # Генерим важный признак для разделения на df[amount>0] и df[amount<0]
    df['amount>0'] = df['amount'].apply(lambda x: int(x > 0))

    robust_scaler = RobustScaler()
    df['amount'] = robust_scaler.fit_transform(df['amount'].values.reshape(1, -1)).reshape(df.shape[0])

    # признаки для времени
    df['TRDATETIME'] = df['TRDATETIME'].map(lambda x: datetime.strptime(x, "%d%b%y:%H:%M:%S"))
    df['day'] = df['TRDATETIME'].map(lambda x: x.day)
    df['weekday'] = df['TRDATETIME'].map(lambda x: x.weekday())
    df['h_date'] = df['TRDATETIME'].map(lambda x: x.hour)

    del df['TRDATETIME']

    # "Базовая фича" - Количество покупок по каждой категории

    X = df.groupby('cl_id') \
        .apply(lambda x: x[['trx_category']].unstack().value_counts()) \
        .unstack() \
        .fillna(0)
    X.rename(columns=lambda x: 'trx_category_value_counts_' + str(x), inplace=True)

    # Сумма покупок по каждой категории
    F2 = df.groupby('cl_id').apply(lambda x: x.groupby('trx_category')['amount'].sum()).unstack().fillna(0)
    F2.rename(columns=lambda x: 'sum_' + str(x), inplace=True)

    # Максимальная покупока по каждой категории

    F3 = df.groupby('cl_id').apply(lambda x: x.groupby('trx_category')['amount'].max()).unstack().fillna(0)
    F3.rename(columns=lambda x: 'max_' + str(x), inplace=True)

    # Дисперсия покупока по каждой категории

    F4 = df.groupby('cl_id').apply(lambda x: x.groupby('trx_category')['amount'].std()).unstack().fillna(0)
    F4.rename(columns=lambda x: 'std_' + str(x), inplace=True)

    # Средняя покупка по каждой категории

    F5 = df.groupby('cl_id').apply(lambda x: x.groupby('trx_category')['amount'].mean()).unstack().fillna(0)
    F5.rename(columns=lambda x: 'mean_' + str(x), inplace=True)

    # Количество trx_category
    F6 = df.groupby('cl_id') \
        .apply(lambda x: x[['trx_category']].unstack().value_counts()) \
        .unstack() \
        .fillna(0)
    F6.rename(columns=lambda x: 'trx_category_value_counts_' + str(x), inplace=True)

    # Сумма покупок по каждой trx_category
    F7 = df.groupby('cl_id').apply(lambda x: x.groupby('trx_category')['amount'].sum()).unstack().fillna(0)
    F7.rename(columns=lambda x: 'trx_category_sum_' + str(x), inplace=True)

    # Max покупока по каждой trx_category
    F8 = df.groupby('cl_id').apply(lambda x: x.groupby('trx_category')['amount'].max()).unstack().fillna(0)
    F8.rename(columns=lambda x: 'trx_category_max_' + str(x), inplace=True)

    # Std покупока по каждой trx_category
    F9 = df.groupby('cl_id').apply(lambda x: x.groupby('trx_category')['amount'].std()).unstack().fillna(0)
    F9.rename(columns=lambda x: 'trx_category_std_' + str(x), inplace=True)

    # Mean покупока по каждой trx_category
    F10 = df.groupby('cl_id').apply(lambda x: x.groupby('trx_category')['amount'].mean()).unstack().fillna(0)
    F10.rename(columns=lambda x: 'trx_category_mean_' + str(x), inplace=True)

    # Сколько дней клиент
    F11 = pd.DataFrame({
        'customer_exp_days': df.groupby('cl_id')['day'].apply(lambda x: max(x) - min(x))
    })

    # Количество покупок по дням недели
    F12 = df.groupby('cl_id').apply(lambda x: x.groupby('weekday')['amount'].mean()).unstack().fillna(0)
    F12.rename(columns=lambda x: 'mean_df_count_by_week_day_' + str(x), inplace=True)

    # Транзакции по часам
    F13 = df.groupby('cl_id').apply(lambda x: x.groupby('h_date')['amount'].mean()).unstack().fillna(0)
    F13.rename(columns=lambda x: 'mean_df_count_by_h_' + str(x), inplace=True)

    # выделяем на основе зануления
    df_pos = df.copy()
    df_pos['amount'] = df_pos['amount'] * df_pos['amount>0']

    df_neg = df.copy()
    df_neg['amount'] = df_neg['amount'] * df_neg['amount>0'].apply(lambda x: abs(x - 1))
    
    #Признаки (*) для df_pos и df_neg
    FP1 = df_pos.groupby('cl_id') \
        .apply(lambda x: x[['MCC']].unstack().value_counts()) \
        .unstack() \
        .fillna(0)
    FP1.rename(columns=lambda x: 'FP_MCC_value_counts_' + str(x), inplace=True)

    # Сумма покупок по каждой категории
    FP2 = df_pos.groupby('cl_id').apply(
        lambda x: x.groupby('MCC')['amount'].sum()).unstack().fillna(0)
    FP2.rename(columns=lambda x: 'FP_sum_' + str(x), inplace=True)

    # Максимальная покупока по каждой категории

    FP3 = df_pos.groupby('cl_id').apply(
        lambda x: x.groupby('MCC')['amount'].max()).unstack().fillna(0)
    FP3.rename(columns=lambda x: 'FP_max_' + str(x), inplace=True)

    # Дисперсия покупока по каждой категории

    FP4 = df_pos.groupby('cl_id').apply(
        lambda x: x.groupby('MCC')['amount'].std()).unstack().fillna(0)
    FP4.rename(columns=lambda x: 'FP_std_' + str(x), inplace=True)

    # Средняя покупка по каждой категории

    FP5 = df_pos.groupby('cl_id').apply(
        lambda x: x.groupby('MCC')['amount'].mean()).unstack().fillna(0)
    FP5.rename(columns=lambda x: 'FP_mean_' + str(x), inplace=True)

    # Количество trx_category
    FP6 = df_pos.groupby('cl_id') \
        .apply(lambda x: x[['trx_category']].unstack().value_counts()) \
        .unstack() \
        .fillna(0)
    FP6.rename(columns=lambda x: 'FP_trx_category_value_counts_' + str(x), inplace=True)

    # Сумма покупок по каждой trx_category
    FP7 = df_pos.groupby('cl_id').apply(
        lambda x: x.groupby('trx_category')['amount'].sum()).unstack().fillna(0)
    FP7.rename(columns=lambda x: 'FP_trx_category_sum_' + str(x), inplace=True)

    # Max покупока по каждой trx_category
    FP8 = df_pos.groupby('cl_id').apply(
        lambda x: x.groupby('trx_category')['amount'].max()).unstack().fillna(0)
    FP8.rename(columns=lambda x: 'FP_trx_category_max_' + str(x), inplace=True)

    # Std покупока по каждой trx_category
    FP9 = df_pos.groupby('cl_id').apply(
        lambda x: x.groupby('trx_category')['amount'].std()).unstack().fillna(0)
    FP9.rename(columns=lambda x: 'FP_trx_category_std_' + str(x), inplace=True)

    # Mean покупока по каждой trx_category
    FP10 = df_pos.groupby('cl_id').apply(
        lambda x: x.groupby('trx_category')['amount'].mean()).unstack().fillna(0)
    FP10.rename(columns=lambda x: 'FP_trx_category_mean_' + str(x), inplace=True)

    # Сколько дней клиент
    FP11 = pd.DataFrame({
        'FP_customer_exp_days': df_pos.groupby('cl_id')['day'].apply(lambda x: max(x) - min(x))
    })

    # Количество покупок по дням недели
    FP12 = df_pos.groupby('cl_id').apply(
        lambda x: x.groupby('weekday')['amount'].mean()).unstack().fillna(0)
    FP12.rename(columns=lambda x: 'FP_mean_df_pos_count_by_week_day_' + str(x), inplace=True)

    # Транзакции по часам
    FP13 = df_pos.groupby('cl_id').apply(
        lambda x: x.groupby('h_date')['amount'].mean()).unstack().fillna(0)
    FP13.rename(columns=lambda x: 'FP_mean_df_pos_count_by_h_' + str(x), inplace=True)

    FN1 = df_neg.groupby('cl_id') \
        .apply(lambda x: x[['MCC']].unstack().value_counts()) \
        .unstack() \
        .fillna(0)
    FN1.rename(columns=lambda x: 'FN_MCC_value_counts_' + str(x), inplace=True)

    # Сумма покупок по каждой категории
    FN2 = df_neg.groupby('cl_id').apply(
        lambda x: x.groupby('MCC')['amount'].sum()).unstack().fillna(0)
    FN2.rename(columns=lambda x: 'FN_sum_' + str(x), inplace=True)

    # Максимальная покупока по каждой категории

    FN3 = df_neg.groupby('cl_id').apply(
        lambda x: x.groupby('MCC')['amount'].max()).unstack().fillna(0)
    FN3.rename(columns=lambda x: 'FN_max_' + str(x), inplace=True)

    # Дисперсия покупок по каждой категории

    FN4 = df_neg.groupby('cl_id').apply(
        lambda x: x.groupby('MCC')['amount'].std()).unstack().fillna(0)
    FN4.rename(columns=lambda x: 'FN_std_' + str(x), inplace=True)

    # Средняя покупка по каждой категории

    FN5 = df_neg.groupby('cl_id').apply(
        lambda x: x.groupby('MCC')['amount'].mean()).unstack().fillna(0)
    FN5.rename(columns=lambda x: 'FN_mean_' + str(x), inplace=True)

    # Количество trx_category
    FN6 = df_neg.groupby('cl_id') \
        .apply(lambda x: x[['trx_category']].unstack().value_counts()) \
        .unstack() \
        .fillna(0)
    FN6.rename(columns=lambda x: 'FN_trx_category_value_counts_' + str(x), inplace=True)

    # Сумма покупок по каждой trx_category
    FN7 = df_neg.groupby('cl_id').apply(
        lambda x: x.groupby('trx_category')['amount'].sum()).unstack().fillna(0)
    FN7.rename(columns=lambda x: 'FN_trx_category_sum_' + str(x), inplace=True)

    # Max покупока по каждой trx_category
    FN8 = df_neg.groupby('cl_id').apply(
        lambda x: x.groupby('trx_category')['amount'].max()).unstack().fillna(0)
    FN8.rename(columns=lambda x: 'FN_trx_category_max_' + str(x), inplace=True)

    # Std покупока по каждой trx_category
    FN9 = df_neg.groupby('cl_id').apply(
        lambda x: x.groupby('trx_category')['amount'].std()).unstack().fillna(0)
    FN9.rename(columns=lambda x: 'FN_trx_category_std_' + str(x), inplace=True)

    # Mean покупока по каждой trx_category
    FN10 = df_neg.groupby('cl_id').apply(
        lambda x: x.groupby('trx_category')['amount'].mean()).unstack().fillna(0)
    FN10.rename(columns=lambda x: 'FN_trx_category_mean_' + str(x), inplace=True)

    # Сколько дней клиент
    FN11 = pd.DataFrame({
        'FN_customer_exp_days': df_neg.groupby('cl_id')['day'].apply(lambda x: max(x) - min(x))
    })

    # Количество покупок по дням недели
    FN12 = df_neg.groupby('cl_id').apply(
        lambda x: x.groupby('weekday')['amount'].mean()).unstack().fillna(0)
    FN12.rename(columns=lambda x: 'FN_mean_df_neg_count_by_week_day_' + str(x), inplace=True)

    # Транзакции по часам
    FN13 = df_neg.groupby('cl_id').apply(
        lambda x: x.groupby('h_date')['amount'].mean()).unstack().fillna(0)
    FN13.rename(columns=lambda x: 'FN_mean_df_neg_count_by_h_' + str(x), inplace=True)

    #Сливаем все группы признаков в один DataFrame
    XFeatured = pd.concat(
        [X, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, FP1, FP2, FP3, FP4, FP5, FP6, FP7, FP8, FP9, FP10, FP11,
         FP12, FP13, FN1, FN2, FN3, FN4, FN5, FN6, FN7, FN8, FN9, FN10, FN11, FN12, FN13], axis=1, join='inner')
    del X, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, FP1, FP2, FP3, FP4, FP5, FP6, FP7, FP8, FP9, FP10, FP11, FP12, FP13, FN1, FN2, FN3, FN4, FN5, FN6, FN7, FN8, FN9, FN10, FN11, FN12, FN13

    XFeatured['cl_id'] = df.groupby(['cl_id'])['cl_id'].max()

    return  XFeatured


def write_submission(pred, test_id, path):
    df = pd.DataFrame(columns = ['_ID_','_VAL_'])
    df['_ID_'] = test_id
    df['_VAL_'] = pred
    df.to_csv(path, index=False)

import lightgbm as lgb
def prepLGB(data, labels, IDCol='', fDrop=[]):


    if IDCol != '':
        IDs = data[IDCol]
    else:
        IDs = []

    if fDrop != []:
        data = data.drop(fDrop, axis=1)

    # Create LGB mats
    lData = lgb.Dataset(data, label=labels, free_raw_data=False,
                        feature_name=list(data.columns),
                        categorical_feature='auto')

    return lData, labels, IDs, data
