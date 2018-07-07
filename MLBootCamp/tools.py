import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD

def base_preprocassing(df):
    #Заполненность счетчиков
    df['cnt1_is_not_null'] = df['cnt1'].apply(lambda x: 0 if x == "{}" else 1)
    df['cnt2_is_not_null'] = df['cnt2'].apply(lambda x: 0 if x == "{}" else 1)
    df['cnt3_is_not_null'] = df['cnt3'].apply(lambda x: 0 if x == "{}" else 1)
    #Достанем кол-во счетчиков в признаке
    df['count_cnt1'] = df['cnt1'].map(lambda x: len(list(json.loads(x).values())))
    df['count_cnt2'] = df['cnt2'].map(lambda x: len(list(json.loads(x).values())))
    df['count_cnt3'] = df['cnt3'].map(lambda x: len(list(json.loads(x).values())))
    #Сумма счетчиков в признаке
    df['sum_cnt1'] = df['cnt1'].map(lambda x: np.sum(list(json.loads(x).values())))
    df['sum_cnt2'] = df['cnt2'].map(lambda x: np.sum(list(json.loads(x).values())))
    df['sum_cnt3'] = df['cnt3'].map(lambda x: np.sum(list(json.loads(x).values())))

    #Группируем данные
    df_base = pd.DataFrame()
    df_base['max_data_diff'] = df.groupby(['cuid'])['data_diff'].max()
    df_base['min_data_diff'] = df.groupby(['cuid'])['data_diff'].min()
    df_base['mean_data_diff'] = df.groupby(['cuid'])['data_diff'].mean()
    df_base['delta_data_diff'] = df.groupby(['cuid'])['data_diff'].max() - df.groupby(['cuid'])['data_diff'].min()

    #Кол-во счетчиков в признаке
    df_base['max_count_cnt1'] = df.groupby(['cuid'])['count_cnt1'].max()
    df_base['max_count_cnt2'] = df.groupby(['cuid'])['count_cnt2'].max()
    df_base['max_count_cnt3'] = df.groupby(['cuid'])['count_cnt3'].max()
    df_base['min_count_cnt1'] = df.groupby(['cuid'])['count_cnt1'].min()
    df_base['min_count_cnt2'] = df.groupby(['cuid'])['count_cnt2'].min()
    df_base['min_count_cnt3'] = df.groupby(['cuid'])['count_cnt3'].min()
    df_base['mean_count_cnt1'] = df.groupby(['cuid'])['count_cnt1'].mean()
    df_base['mean_count_cnt2'] = df.groupby(['cuid'])['count_cnt2'].mean()
    df_base['mean_count_cnt3'] = df.groupby(['cuid'])['count_cnt3'].mean()
    #Сумма счетчиков в признаке
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

    return df_base

def dict_vect_preprocessing(df):
    df = df.sort_values(['data_diff'], ascending = True)
    df_dict = pd.DataFrame()
    df_dict['cuid'] = df.groupby(['cuid'])['cuid'].min()
    df_dict['target'] = df.groupby(['cuid'])['target'].min()

    df['cnt1'] = df['cnt1'].apply(lambda x: x[1:-1]+',' if len(x)>2 else '')
    df['cnt2'] = df['cnt2'].apply(lambda x: x[1:-1]+',' if len(x)>2 else '')
    df['cnt3'] = df['cnt3'].apply(lambda x: x[1:-1]+',' if len(x)>2 else '')

    df_dict['cnt1'] = df.groupby(['cuid'])['cnt1'].sum().apply(lambda x: x[:-1] if (len(x)>0 and x[-1]==',') else x)
    df_dict['cnt1'] = df_dict['cnt1'].apply(lambda x: json.loads('{'+x+'}'))

    df_dict['cnt2'] = df.groupby(['cuid'])['cnt2'].sum().apply(lambda x: x[:-1] if (len(x)>0 and x[-1]==',') else x)
    df_dict['cnt2'] = df_dict['cnt2'].apply(lambda x: json.loads('{'+x+'}'))

    df_dict['cnt3'] = df.groupby(['cuid'])['cnt3'].sum().apply(lambda x: x[:-1] if (len(x)>0 and x[-1]==',') else x)
    df_dict['cnt3'] = df_dict['cnt3'].apply(lambda x: json.loads('{'+x+'}'))

    dv = DictVectorizer(separator=':')
    dv_cnt1 = dv.fit_transform(df_dict['cnt1'])
    dv_cnt2 = dv.fit_transform(df_dict['cnt2'])
    dv_cnt3 = dv.fit_transform(df_dict['cnt3'])

    svd = TruncatedSVD(n_components=100, random_state=17, n_iter=5)
    dv_cnt1 = svd.fit_transform(dv_cnt1)
    dv_cnt2 = svd.fit_transform(dv_cnt2)
    dv_cnt3 = svd.fit_transform(dv_cnt3)

    return pd.DataFrame(np.hstack([df_dict[['cuid','target']].values.reshape(-1,2),dv_cnt1,dv_cnt2,dv_cnt3]), columns=['cuid','target']+list(range(dv_cnt1.shape[1]*3)))

def tfidf_preprocessing(df):
    #Номер счетчика + значение
    df = df.sort_values(['data_diff'], ascending = True)
    df_tfidf = pd.DataFrame()
    df_tfidf['cuid'] = df.groupby(['cuid'])['cuid'].min()
    df_tfidf['target'] = df.groupby(['cuid'])['target'].min()

    df['cnt1'] = df['cnt1'].apply(lambda x: x[1:-1])
    df['cnt2'] = df['cnt2'].apply(lambda x: x[1:-1])
    df['cnt3'] = df['cnt3'].apply(lambda x: x[1:-1])

    tokenizer = lambda doc: doc[1:-1].split(',')
    tf_idf = TfidfVectorizer(tokenizer = tokenizer, max_df = 0.90, min_df = 0.01, ngram_range=(1,1))

    new_df_cnt1 = df.groupby(['cuid'])['cnt1'].sum()
    new_df_cnt2 = df.groupby(['cuid'])['cnt2'].sum()
    new_df_cnt3 = df.groupby(['cuid'])['cnt3'].sum()

    tf_idf_cnt1 = tf_idf.fit_transform(new_df_cnt1)
    tf_idf_cnt2 = tf_idf.fit_transform(new_df_cnt2)
    tf_idf_cnt3 = tf_idf.fit_transform(new_df_cnt3)

    svd = TruncatedSVD(n_components=150, random_state=17, n_iter=5)
    tf_idf_cnt1 = svd.fit_transform(tf_idf_cnt1)
    tf_idf_cnt2 = svd.fit_transform(tf_idf_cnt2)
    #tf_idf_cnt3 = svd.fit_transform(tf_idf_cnt3)

    return pd.DataFrame(np.hstack([df_tfidf[['cuid','target']].values.reshape(-1,2),tf_idf_cnt1,tf_idf_cnt2]), columns=['cuid','target']+list(range(tf_idf_cnt1.shape[1]*2)))
