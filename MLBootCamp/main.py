import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Normalizer, MinMaxScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.decomposition import TruncatedSVD, PCA
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from tools import base_preprocassing, dict_vect_preprocessing, tfidf_preprocessing, split_train_test

ACTION = 'cloud_dict'
PATH = './data/'



if ACTION=='train':

    print('read data')
    #df = pd.read_csv(os.path.join(PATH, 'train_sample.csv'), nrows=1000000)

    df = pd.read_csv(os.path.join(PATH,'mlboot_data.tsv'), delimiter='\t', nrows=2000000, header=None)
    df.columns = ['cuid', 'cat', 'cnt1','cnt2','cnt3','date_diff']
    df = df.join(pd.read_csv(os.path.join(PATH,'mlboot_train_answers.tsv'), delimiter='\t').set_index('cuid'), on='cuid', how='inner')

    print('preprocess')
    ohe_cat = OneHotEncoder(sparse=False)
    df, ohe_cat = base_preprocassing(df, ohe_cat)
    df = df.convert_objects(convert_numeric=True)

    print('main part')
    X_train, X_test, y_train, y_test = train_test_split(df.drop(axis=1, columns=['cuid','target']),df.target, test_size=0.2, random_state=17)

    # pca = PCA(n_components=15)
    n = StandardScaler()
    X_train = n.fit_transform(X_train)
    X_test = n.transform(X_test)

    # lr = LogisticRegression(penalty='l2', C=0.3, solver='lbfgs')
    # sbfs = SFS(lr,
    #            k_features=11,
    #            forward=False,
    #            floating=True,
    #            scoring='roc_auc',
    #            cv=5,
    #            n_jobs=-1,
    #            verbose=1)
    #
    # sbfs = sbfs.fit(X_train, y_train)
    #
    # print('\nSequential Backward Floating Selection (k=3):')
    # print(sbfs.k_feature_idx_)
    # print('CV Score:')
    # print(sbfs.k_score_)
    #
    # # Index(['1_sum', '2_sum', '5_sum', '3_min', 'max_date_diff', 'max_count_cnt3',
    # #        'mean_sum_cnt1', 'mean_sum_cnt2', 'mean_sum_cnt3', 'cat_min_le'],dtype='object')
    #
    # X_train, X_test, y_train, y_test = train_test_split(df[df.columns[[[*map(lambda x: x+2,sbfs.k_feature_idx_)]]]], df.target,
    #                                                     test_size=0.2, random_state=17)
    #
    # n = StandardScaler()
    # X_train = n.fit_transform(X_train)
    # X_test = n.transform(X_test)

    lr = LogisticRegression(penalty='l2', C=0.2, solver='lbfgs')
    # lr = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio = 0.15, random_state=17,class_weight={0:0.4, 1:0.6})
    sf = StratifiedKFold(n_splits=10, shuffle=True, random_state=17)
    cv = cross_val_score(lr,X_train, y_train, scoring='roc_auc', cv=sf, n_jobs=-1)
    print (cv.mean(), cv.std())
    lr.fit(X_train,y_train)
    pred = lr.predict_proba(X_test)
    print (roc_auc_score(y_test,pred[:, 1]))

    # from evolutionary_search import EvolutionaryAlgorithmSearchCV
    # paramgrid = {"solver": ["liblinear", 'lbfgs'],
    #              "C"     : np.arange(0.1,1,0.1),
    #              "penalty" : ['l2']}
    #
    # cv = EvolutionaryAlgorithmSearchCV(estimator=LogisticRegression(),
    #                                    params=paramgrid,
    #                                    scoring="roc_auc",
    #                                    cv=StratifiedKFold(n_splits=5),
    #                                    verbose=1,
    #                                    population_size=100,
    #                                    gene_mutation_prob=0.10,
    #                                    gene_crossover_prob=0.5,
    #                                    tournament_size=3,
    #                                    generations_number=5,
    #                                    n_jobs=4)
    # cv.fit(X_train, y_train)
    # Best individual is: {'solver': 'lbfgs', 'C': 0.2, 'penalty': 'l2'}
    # with fitness: 0.6307805995054357

    #SGDClassifier
elif ACTION == 'predict':
    train_ans = pd.read_csv(os.path.join(PATH,'mlboot_train_answers.tsv'), delimiter='\t')
    test_id = pd.read_csv(os.path.join(PATH,'mlboot_test.tsv'), delimiter='\t')

    step = 3100000 #19528597
    for part, nrow in tqdm(enumerate(range(0,19528597,step))):
        df = pd.read_csv(os.path.join(PATH,'mlboot_data.tsv'), delimiter='\t',nrows=step, skiprows=range(0,nrow),header=None)
        df.columns = ['cuid', 'cat', 'cnt1','cnt2','cnt3','data_diff']

        train_df = df.join(train_ans.set_index('cuid'), on='cuid', how='inner')
        test_df = df.join(test_id.set_index('cuid'), on='cuid', how='inner')

        train_df = dict_vect_preprocessing(train_df)
        train_df = train_df.convert_objects(convert_numeric=True)

        test_df = dict_vect_preprocessing(test_df, is_train=False)
        test_df = test_df.convert_objects(convert_numeric=True)

        X_train = train_df.drop(axis=1, columns=['cuid','target'])
        y_train = train_df['target']
        del train_df

        X_test = test_df.drop(axis=1, columns=['cuid'])
        test_id_part = test_df['cuid']
        del test_df

        n = Normalizer()
        X_train = n.fit_transform(X_train)
        X_test = n.transform(X_test)

        lr = LogisticRegression(penalty='l2', C=0.2,solver='lbfgs')
        lr.fit(X_train,y_train)

        scaler = MinMaxScaler()
        ans_df = pd.DataFrame(test_id_part, columns=['cuid'])
        ans_df['target'] = scaler.fit_transform(lr.predict_proba(X_test)[:, 1].reshape(-1,1))

        ans_df.to_csv(os.path.join('submission','submission_part{}.csv'.format(part)), index=False)
elif ACTION == 'union':
    test_id = pd.read_csv(os.path.join(PATH,'mlboot_test.tsv'), delimiter='\t')
    union_df = pd.DataFrame(columns=['cuid','target'])
    path = './submission'
    files = os.listdir(path)
    for file in files:
        if 'part' in file:
            df = pd.read_csv(os.path.join(path,file))
            union_df=pd.concat([union_df,df[['cuid','target']]],axis=0)
    union_df = union_df.groupby(['cuid'])['target'].mean()
    predicts = test_id.set_index('cuid').join(union_df, on='cuid', how='inner')
    predicts.to_csv(os.path.join(path,'union.csv'), index=False, header=False)
elif ACTION == 'cloud_dict':
    test_id = pd.read_csv(os.path.join(PATH,'mlboot_test.tsv'), delimiter='\t')
    # print ('read data')
    # df = pd.read_csv(os.path.join(PATH,'train.csv'), nrows=7000000)
    #
    # print ('preprocessing')
    # dict1 = DictVectorizer(separator=':')
    # dict2 = DictVectorizer(separator=':')
    # dict3 = DictVectorizer(separator=':')
    #
    # svd1 = TruncatedSVD(n_components=100, random_state=17, n_iter=5)
    # svd2 = TruncatedSVD(n_components=100, random_state=17, n_iter=5)
    # svd3 = TruncatedSVD(n_components=100, random_state=17, n_iter=5)
    #
    # df, dict, svd = dict_vect_preprocessing(df, [dict1,dict2,dict3], [svd1,svd2,svd3])
    #
    # df = df.convert_objects(convert_numeric=True)
    # print ('train')
    # lr = LogisticRegression(penalty='l2', solver='lbfgs', C=0.2)
    # n = Normalizer()
    # lr.fit(n.fit_transform(df.drop(axis=1, columns=['cuid','target'])), df['target'])
    #
    # print ('read test')
    # df = pd.read_csv(os.path.join(PATH,'test.csv'))
    #
    # print ('process test')
    # df, dict, svd = dict_vect_preprocessing(df, dict, svd, is_train=False)
    # df = df.convert_objects(convert_numeric=True)
    #
    # out_df = pd.DataFrame()
    # out_df['cuid'] = df.cuid
    #
    # print ('predict test')
    # df = n.transform(df.drop(axis=1, columns=['cuid']))
    # pred = lr.predict_proba(df)
    # out_df['target'] = pred[:, 1]
    #
    # predicts = test_id.join(out_df.set_index('cuid'), on='cuid', how='inner')
    # predicts['target'].to_csv(os.path.join(PATH,'predict_test_part1.csv'), index=False, header=False)

    ###### PART 2
    print ('read data_part2')
    df = pd.read_csv(os.path.join(PATH,'train.csv'), skiprows=(1,7000000))
    print(df.shape)

    print ('preprocessing_part2')
    dict1 = DictVectorizer(separator=':')
    dict2 = DictVectorizer(separator=':')
    dict3 = DictVectorizer(separator=':')

    svd1 = TruncatedSVD(n_components=100, random_state=17, n_iter=5)
    svd2 = TruncatedSVD(n_components=100, random_state=17, n_iter=5)
    svd3 = TruncatedSVD(n_components=100, random_state=17, n_iter=5)

    df, dict, svd = dict_vect_preprocessing(df, [dict1,dict2,dict3], [svd1,svd2,svd3])
    df = df.convert_objects(convert_numeric=True)

    print ('train_part2')
    lr = LogisticRegression(penalty='l2', solver='lbfgs', C=0.2)
    n = Normalizer()
    lr.fit(n.fit_transform(df.drop(axis=1, columns=['cuid','target'])), df['target'])

    print ('read test_part2')
    df = pd.read_csv(os.path.join(PATH,'test.csv'))

    print ('process test_part2')
    df, dict, svd = dict_vect_preprocessing(df, dict, svd, is_train=False)
    df = df.convert_objects(convert_numeric=True)

    out_df = pd.DataFrame()
    out_df['cuid'] = df.cuid

    print ('predict test_part2')
    df = n.transform(df.drop(axis=1, columns=['cuid']))
    pred = lr.predict_proba(df)
    out_df['target'] = pred[:, 1]

    predicts = test_id.join(out_df.set_index('cuid'), on='cuid', how='inner')
    predicts['target'].to_csv(os.path.join(PATH,'predict_test_part2.csv'), index=False, header=False)
elif ACTION == 'cloud_tfidf':
    test_id = pd.read_csv(os.path.join(PATH, 'mlboot_test.tsv'), delimiter='\t')
    print ('read data')
    df = pd.read_csv(os.path.join(PATH, 'train.csv'))

    print ('preprocessing')
    tokenizer = lambda doc: doc[1:-1].split(',')
    tf_idf1 = TfidfVectorizer(tokenizer=tokenizer, max_df=0.90, min_df=0.01, ngram_range=(1, 1))
    tf_idf2 = TfidfVectorizer(tokenizer=tokenizer, max_df=0.90, min_df=0.01, ngram_range=(1, 1))
    tf_idf3 = TfidfVectorizer(tokenizer=tokenizer, max_df=0.90, min_df=0.01, ngram_range=(1, 1))

    svd1 = TruncatedSVD(n_components=100, random_state=17, n_iter=5)
    svd2 = TruncatedSVD(n_components=100, random_state=17, n_iter=5)
    svd3 = TruncatedSVD(n_components=100, random_state=17, n_iter=5)

    df, tfidf, svd = tfidf_preprocessing(df, [tf_idf1, tf_idf2, tf_idf3], [svd1, svd2, svd3])
    df = df.convert_objects(convert_numeric=True)

    print ('train')
    lr = LogisticRegression(penalty='l2', solver='lbfgs', C=0.2)
    n = Normalizer()
    lr.fit(n.fit_transform(df.drop(axis=1, columns=['cuid', 'target'])), df['target'])

    print ('read test')
    df = pd.read_csv(os.path.join(PATH, 'test.csv'))

    print ('process test')
    df, tfidf, svd = tfidf_preprocessing(df, tfidf, svd, is_train=False)
    df = df.convert_objects(convert_numeric=True)

    out_df = pd.DataFrame()
    out_df['cuid'] = df.cuid

    print ('predict test')
    df = n.transform(df.drop(axis=1, columns=['cuid']))
    pred = lr.predict_proba(df)
    out_df['target'] = pred[:, 1]

    predicts = test_id.join(out_df.set_index('cuid'), on='cuid', how='inner')
    predicts['target'].to_csv(os.path.join(PATH, 'predict_test_tfidf.csv'), index=False, header=False)
elif ACTION == 'ansamblirovat':
    df1 = pd.read_csv(os.path.join(PATH, 'predict_test_dict.csv'), header=None)
    df2 = pd.read_csv(os.path.join(PATH, 'predict_test_tfidf.csv'), header=None)

    union_df = df1[0] * 0.6 + df2[0] * 0.4
    union_df.to_csv(os.path.join(PATH, 'dict_and_tfidf.csv'), index=False, header=False)