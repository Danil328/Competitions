import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Normalizer, MinMaxScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, SGDClassifier

from tools import base_preprocassing, dict_vect_preprocessing, tfidf_preprocessing, split_train_test

ACTION = 'claoud'
PATH = './data/'

if ACTION=='train':

    #read data
    # df = pd.read_csv(os.path.join(PATH, 'train_sample.csv'), nrows=1000000)
    # df = df.join(pd.read_csv(os.path.join(PATH,'mlboot_train_answers.tsv'), delimiter='\t').set_index('cuid'), on='cuid', how='inner')

    df = pd.read_csv(os.path.join(PATH,'mlboot_data.tsv'), delimiter='\t', nrows=3000000, header=None)
    df.columns = ['cuid', 'cat', 'cnt1','cnt2','cnt3','data_diff']
    df = df.join(pd.read_csv(os.path.join(PATH,'mlboot_train_answers.tsv'), delimiter='\t').set_index('cuid'), on='cuid', how='inner')

    #preprocess
    df = dict_vect_preprocessing(df)
    df = df.convert_objects(convert_numeric=True)

    #tr
    X_train, X_test, y_train, y_test = train_test_split(df.drop(axis=1, columns=['cuid','target']),df.target, test_size=0.2, random_state=17)

    n = Normalizer()
    X_train = n.fit_transform(X_train)
    X_test = n.transform(X_test)

    lr = LogisticRegression(penalty='l2', C=0.2,solver='lbfgs')
    #lr = SGDClassifier(loss='log')
    sf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
    cv = cross_val_score(lr,X_train, y_train, scoring='roc_auc', cv=sf, n_jobs=-1)
    print (cv.mean(), cv.std())
    lr.fit(X_train,y_train)
    pred = lr.predict_proba(X_test)
    print (roc_auc_score(y_test,pred[:, 1]))

    # from evolutionary_search import EvolutionaryAlgorithmSearchCV
    #
    # paramgrid = {"solver": ["liblinear", 'lbfgs'],
    #              "C"     : np.arange(0.1,1,0.1),
    #              "penalty" : ['l2']}
    #
    # cv = EvolutionaryAlgorithmSearchCV(estimator=LogisticRegression(),
    #                                    params=paramgrid,
    #                                    scoring="roc_auc",
    #                                    cv=StratifiedKFold(n_splits=5),
    #                                    verbose=1,
    #                                    population_size=50,
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
elif ACTION == 'cloud':
    test_id = pd.read_csv(os.path.join(PATH,'mlboot_test.tsv'), delimiter='\t')
    print ('read data')
    df = pd.read_csv(os.path.join(PATH,'train.csv'), nrows=7000000)

    print ('preprocessing')
    df = dict_vect_preprocessing(df)
    df = df.convert_objects(convert_numeric=True)
    print ('train')
    lr = LogisticRegression(penalty='l2', solver='lbfgs', C=0.2)
    n = Normalizer()
    lr.fit(n.fit_transform(df.drop(axis=1, columns=['cuid','target'])), df['target'])

    print ('read test')
    df = pd.read_csv(os.path.join(PATH,'test.csv'))

    print ('process test')
    df = dict_vect_preprocessing(df, is_train=False)
    df = df.convert_objects(convert_numeric=True)

    out_df = pd.DataFrame()
    out_df['cuid'] = df.cuid

    print ('predict test')
    df = n.transform(df.drop(axis=1, columns=['cuid']))
    pred = lr.predict_proba(df)
    out_df['target'] = pred[:, 1]

    predicts = test_id.join(out_df.set_index('cuid'), on='cuid', how='inner')
    predicts['target'].to_csv(os.path.join(PATH,'predict_test.csv'), index=False, header=False)

