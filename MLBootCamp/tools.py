import pandas as pd
import numpy as np
import json
import os
import gc
from tqdm import tqdm

gc.enable()


def base_preprocassing(df, ohe_cat):
    # Работаем с категориальным признаком
    cat_ohe = ohe_cat.fit_transform(df['cat'].values.reshape(-1,1))
    cat_ohe = pd.DataFrame(np.hstack((df.cuid.values.reshape(-1,1), cat_ohe)), columns=['cuid']+list(ohe_cat.active_features_))
    cat_ohe = cat_ohe.convert_objects(convert_numeric=True)

    # Заполненность счетчиков
    df['cnt1_is_not_null'] = df['cnt1'].apply(lambda x: 0 if x == "{}" else 1)
    df['cnt2_is_not_null'] = df['cnt2'].apply(lambda x: 0 if x == "{}" else 1)
    df['cnt3_is_not_null'] = df['cnt3'].apply(lambda x: 0 if x == "{}" else 1)
    # Достанем кол-во счетчиков в признаке
    df['count_cnt1'] = df['cnt1'].map(lambda x: len(list(json.loads(x).values())))
    df['count_cnt2'] = df['cnt2'].map(lambda x: len(list(json.loads(x).values())))
    df['count_cnt3'] = df['cnt3'].map(lambda x: len(list(json.loads(x).values())))
    # Сумма счетчиков в признаке
    df['sum_cnt1'] = df['cnt1'].map(lambda x: np.sum(list(json.loads(x).values())))
    df['sum_cnt2'] = df['cnt2'].map(lambda x: np.sum(list(json.loads(x).values())))
    df['sum_cnt3'] = df['cnt3'].map(lambda x: np.sum(list(json.loads(x).values())))

    # Группируем данные
    df_base = pd.DataFrame()
    df_base['cuid'] = df.groupby(['cuid'])['cuid'].min()
    df_base['target'] = df.groupby(['cuid'])['target'].min()
    # cat
    df_base[[*map(lambda x: str(x)+'mean',list(ohe_cat.active_features_))]] = cat_ohe.groupby(['cuid'])[list(ohe_cat.active_features_)].mean() #0.55427
    df_base[[*map(lambda x: str(x)+'_sum',list(ohe_cat.active_features_))]] = cat_ohe.groupby(['cuid'])[list(ohe_cat.active_features_)].sum() #0.56613
    df_base[[*map(lambda x: str(x) + '_min', list(ohe_cat.active_features_))]] = cat_ohe.groupby(['cuid'])[list(ohe_cat.active_features_)].min() #0.54849
    df_base[[*map(lambda x: str(x) + '_max', list(ohe_cat.active_features_))]] = cat_ohe.groupby(['cuid'])[list(ohe_cat.active_features_)].max() #0.55440

    df_base['max_date_diff'] = df.groupby(['cuid'])['date_diff'].max() #0.5183
    df_base['min_date_diff'] = df.groupby(['cuid'])['date_diff'].min()
    df_base['mean_date_diff'] = df.groupby(['cuid'])['date_diff'].mean()
    df_base['delta_date_diff'] = df.groupby(['cuid'])['date_diff'].max() - df.groupby(['cuid'])['date_diff'].min()

    # Кол-во счетчиков в признаке
    df_base['max_count_cnt1'] = df.groupby(['cuid'])['count_cnt1'].max()
    df_base['max_count_cnt2'] = df.groupby(['cuid'])['count_cnt2'].max()
    df_base['max_count_cnt3'] = df.groupby(['cuid'])['count_cnt3'].max()
    df_base['min_count_cnt1'] = df.groupby(['cuid'])['count_cnt1'].min()
    df_base['min_count_cnt2'] = df.groupby(['cuid'])['count_cnt2'].min()
    df_base['min_count_cnt3'] = df.groupby(['cuid'])['count_cnt3'].min()
    df_base['mean_count_cnt1'] = df.groupby(['cuid'])['count_cnt1'].mean()
    df_base['mean_count_cnt2'] = df.groupby(['cuid'])['count_cnt2'].mean()
    df_base['mean_count_cnt3'] = df.groupby(['cuid'])['count_cnt3'].mean()
    # Сумма счетчиков в признаке
    df_base['max_sum_cnt1'] = df.groupby(['cuid'])['sum_cnt1'].max()
    df_base['max_sum_cnt2'] = df.groupby(['cuid'])['sum_cnt2'].max()
    df_base['max_sum_cnt3'] = df.groupby(['cuid'])['sum_cnt3'].max()
    df_base['min_sum_cnt1'] = df.groupby(['cuid'])['sum_cnt1'].min()
    df_base['min_sum_cnt2'] = df.groupby(['cuid'])['sum_cnt2'].min()
    df_base['min_sum_cnt3'] = df.groupby(['cuid'])['sum_cnt3'].min()
    df_base['mean_sum_cnt1'] = df.groupby(['cuid'])['sum_cnt1'].mean()
    df_base['mean_sum_cnt2'] = df.groupby(['cuid'])['sum_cnt2'].mean()
    df_base['mean_sum_cnt3'] = df.groupby(['cuid'])['sum_cnt3'].mean()

    df_base['cat_min_le'] = df.groupby(['cuid'])['cat'].min()

    # Кол-во rows
    df_base['cound_rows'] = df.groupby(['cuid'])['cuid'].count()

    return df_base, ohe_cat


def dict_vect_preprocessing(df, dict, svd, is_train=True):
    df = df.sort_values(['date_diff'], ascending=True)
    df_dict = pd.DataFrame()
    df_dict['cuid'] = df.groupby(['cuid'])['cuid'].min()
    if is_train:
        df_dict['target'] = df.groupby(['cuid'])['target'].min()
        cols = ['cuid', 'target']
    else:
        cols = ['cuid']

    df['cnt1'] = df['cnt1'].apply(lambda x: x[1:-1] + ',' if len(x) > 2 else '')
    df['cnt2'] = df['cnt2'].apply(lambda x: x[1:-1] + ',' if len(x) > 2 else '')
    df['cnt3'] = df['cnt3'].apply(lambda x: x[1:-1] + ',' if len(x) > 2 else '')

    df_dict['cnt1'] = df.groupby(['cuid'])['cnt1'].sum().apply(lambda x: x[:-1] if (len(x) > 0 and x[-1] == ',') else x)
    df_dict['cnt1'] = df_dict['cnt1'].apply(lambda x: json.loads('{' + x + '}'))

    df_dict['cnt2'] = df.groupby(['cuid'])['cnt2'].sum().apply(lambda x: x[:-1] if (len(x) > 0 and x[-1] == ',') else x)
    df_dict['cnt2'] = df_dict['cnt2'].apply(lambda x: json.loads('{' + x + '}'))

    df_dict['cnt3'] = df.groupby(['cuid'])['cnt3'].sum().apply(lambda x: x[:-1] if (len(x) > 0 and x[-1] == ',') else x)
    df_dict['cnt3'] = df_dict['cnt3'].apply(lambda x: json.loads('{' + x + '}'))

    if is_train:
        dv_cnt1 = dict[0].fit_transform(df_dict['cnt1'])
        dv_cnt2 = dict[1].fit_transform(df_dict['cnt2'])
        dv_cnt3 = dict[2].fit_transform(df_dict['cnt3'])

        dv_cnt1 = svd[0].fit_transform(dv_cnt1)
        dv_cnt2 = svd[1].fit_transform(dv_cnt2)
        dv_cnt3 = svd[2].fit_transform(dv_cnt3)
    else:
        dv_cnt1 = dict[0].transform(df_dict['cnt1'])
        dv_cnt2 = dict[1].transform(df_dict['cnt2'])
        dv_cnt3 = dict[2].transform(df_dict['cnt3'])

        dv_cnt1 = svd[0].transform(dv_cnt1)
        dv_cnt2 = svd[1].transform(dv_cnt2)
        dv_cnt3 = svd[2].transform(dv_cnt3)

    return pd.DataFrame(np.hstack([df_dict[cols].values.reshape(-1, len(cols)), dv_cnt1, dv_cnt2, dv_cnt3]),
                        columns=cols + list(range(dv_cnt1.shape[1] + dv_cnt2.shape[1] + dv_cnt3.shape[1]))), dict, svd


def tfidf_preprocessing(df, tfidf, svd, is_train=True):
    # Номер счетчика + значение
    df = df.sort_values(['date_diff'], ascending=True)
    df_tfidf = pd.DataFrame()
    df_tfidf['cuid'] = df.groupby(['cuid'])['cuid'].min()

    if is_train:
        df_tfidf['target'] = df.groupby(['cuid'])['target'].min()
        cols = ['cuid', 'target']
    else:
        cols = ['cuid']

    df['cnt1'] = df['cnt1'].apply(lambda x: x[1:-1])
    df['cnt2'] = df['cnt2'].apply(lambda x: x[1:-1])
    df['cnt3'] = df['cnt3'].apply(lambda x: x[1:-1])

    new_df_cnt1 = df.groupby(['cuid'])['cnt1'].sum()
    new_df_cnt2 = df.groupby(['cuid'])['cnt2'].sum()
    new_df_cnt3 = df.groupby(['cuid'])['cnt3'].sum()
    if is_train:
        new_df_cnt1 = tfidf[0].fit_transform(new_df_cnt1)
        new_df_cnt2 = tfidf[1].fit_transform(new_df_cnt2)
        new_df_cnt3 = tfidf[2].fit_transform(new_df_cnt3)

        tf_idf_cnt1 = svd[0].fit_transform(new_df_cnt1)
        tf_idf_cnt2 = svd[1].fit_transform(new_df_cnt2)
        # tf_idf_cnt3 = svd[2].fit_transform(new_df_cnt3)
    else:
        new_df_cnt1 = tfidf[0].transform(new_df_cnt1)
        new_df_cnt2 = tfidf[1].transform(new_df_cnt2)
        new_df_cnt3 = tfidf[2].transform(new_df_cnt3)

        tf_idf_cnt1 = svd[0].transform(new_df_cnt1)
        tf_idf_cnt2 = svd[1].transform(new_df_cnt2)
        # tf_idf_cnt3 = svd[2].transform(new_df_cnt3)

    return pd.DataFrame(np.hstack([df_tfidf[cols].values.reshape(-1, len(cols)), tf_idf_cnt1, tf_idf_cnt2]),
                        columns=cols + list(range(tf_idf_cnt1.shape[1] * 2))), tfidf, svd

def split_train_test(PATH):
    # split train and test
    train_ans = pd.read_csv(os.path.join(PATH, 'mlboot_train_answers.tsv'), delimiter='\t')
    # test_id = pd.read_csv(os.path.join(PATH,'mlboot_test.tsv'), delimiter='\t')
    step = 3000000

    # test_all = pd.DataFrame(columns = ['cuid', 'cat', 'cnt1','cnt2','cnt3','date_diff'])
    train_all = pd.DataFrame(columns=['cuid', 'cat', 'cnt1', 'cnt2', 'cnt3', 'date_diff', 'target'])
    for part, nrow in tqdm(enumerate(range(0, 19528597, step))):
        df = pd.read_csv(os.path.join(PATH, 'mlboot_data.tsv'), delimiter='\t', nrows=step, skiprows=range(0, nrow),
                         header=None)
        df.columns = ['cuid', 'cat', 'cnt1', 'cnt2', 'cnt3', 'date_diff']

        # test_df = df.join(test_id.set_index('cuid'), on='cuid', how='inner')
        train_df = df.join(train_ans.set_index('cuid'), on='cuid', how='inner')

        # test_all = pd.concat([test_all,test_df])
        train_all = pd.concat([train_all, train_df])

    # test_all.to_csv(os.path.join(PATH,'test.csv'),index=False)
    train_all.to_csv(os.path.join(PATH, 'train.csv'), index=False)