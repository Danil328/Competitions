import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, tqdm_notebook

from tools import write_submission, preprocessing, preprocessing_sber,prepLGB
from imblearn.over_sampling import ADASYN, SMOTE
from currency_converter import CurrencyConverter
from datetime import date

# load data
path_to_data = 'data/'
train = pd.read_csv(os.path.join(path_to_data, 'train.csv'))
test = pd.read_csv(os.path.join(path_to_data, 'test.csv'))
target = train.groupby(['cl_id'])[['target_flag', 'target_sum']].max()
train_cl_id = train.groupby(['cl_id'])['cl_id'].max()
test_cl_id = test.groupby(['cl_id'])['cl_id'].max()
df = pd.concat([train.drop(columns=['target_flag', 'target_sum'], axis=1), test])

MODEL = 'LGB'
PREPROCESSING = 'MY'

if MODEL == 'LGB':
    print ('preprocessing')
    df = preprocessing(df)

    X_train = df.iloc[train_cl_id]
    X_test = df.iloc[test_cl_id]

    X_train.drop(columns=['cl_id'], axis=1, inplace = True)
    X_test.drop(columns=['cl_id'], axis=1, inplace = True)

    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    # smote = SMOTE(kind='borderline1', random_state=17)
    # X_train,y_train = smote.fit_sample(X_train,y_train)

    # X_train_sp,X_val,y_train_sp,y_val = train_test_split(X_train, target.target_flag, test_size=0.2, shuffle=True, random_state=17)
    import lightgbm as lgb
    #
    # lgb_train = lgb.Dataset(X_train_sp, y_train_sp)
    # lgb_val = lgb.Dataset(X_val,y_val,reference=lgb_train)
    #
    # params = {
    #     #'max_depth': 8,
    #     'task': 'train',
    #     'boosting_type': 'gbdt',
    #     'objective': 'binary',
    #     'is_unbalance' : True,
    #     'metric': 'auc',
    #     'lambda_l2' : 1,
    #     'num_leaves': 31,
    #     'learning_rate': 0.05,
    #     'feature_fraction': 0.9,
    #     'bagging_fraction': 0.8,
    #     'bagging_freq': 5,
    #     'verbose': 1
    # }
    #
    # print('Start training...')
    # cv_result_lgb = lgb.cv(params,
    #                        lgb_train,
    #                        num_boost_round=100,
    #                        nfold=5,
    #                        stratified=True,
    #                        shuffle=True,
    #                        early_stopping_rounds=21,
    #                        verbose_eval=0,
    #                        show_stdv=True)
    #
    # model_lgb = lgb.train(params, lgb_train, 5, valid_sets=lgb_val, early_stopping_rounds=21)

    # print('Save model...')
    # # save model to file
    # gbm.save_model('model.txt')

    # Create parameters to search
    params = {'boosting_type': 'gbdt',
              'max_depth': -1,
              'objective': 'binary',
              'nthread': 5,  # Updated from nthread
              'num_leaves': 64,
              'learning_rate': 0.05,
              'max_bin': 512,
              'subsample_for_bin': 200,
              'subsample': 1,
              'subsample_freq': 1,
              'colsample_bytree': 0.8,
              'reg_alpha': 5,
              'reg_lambda': 10,
              'min_split_gain': 0.5,
              'min_child_weight': 1,
              'min_child_samples': 5,
              'scale_pos_weight': 1,
              'num_class': 1,
              'metric': 'binary_error'}

    gridParams = {
        'learning_rate': [0.005],
        #'n_estimators': [8, 16, 24],
        #'num_leaves': [6, 8, 12, 16],
        'boosting_type': ['gbdt'],
        'objective': ['binary'],
        'random_state': [17],  # Updated from 'seed'
        #'colsample_bytree': [0.64, 0.65, 0.66],
        #'subsample': [0.7, 0.75],
        #'reg_alpha': [1, 1.2],
        #'reg_lambda': [1, 1.2, 1.4],
    }
    mdl = lgb.LGBMClassifier(boosting_type='gbdt',
                             objective='binary',
                             silent=True,
                             max_depth=params['max_depth'],
                             max_bin=params['max_bin'],
                             subsample_for_bin=params['subsample_for_bin'],
                             subsample=params['subsample'],
                             subsample_freq=params['subsample_freq'],
                             min_split_gain=params['min_split_gain'],
                             min_child_weight=params['min_child_weight'],
                             min_child_samples=params['min_child_samples'],
                             scale_pos_weight=params['scale_pos_weight'])

    # To view the default model params:
    mdl.get_params().keys()

    # Create the grid
    grid = GridSearchCV(mdl, gridParams, verbose=1, cv=4, n_jobs=-1)
    # Run the grid
    grid.fit(X_train, target.target_flag)

    # Print the best parameters found
    print(grid.best_params_)
    print(grid.best_score_)

    # Kit k models with early-stopping on different training/validation splits

    # k = 12;
    # predsValid = 0
    # predsTrain = 0
    # predsTest = 0
    # for i in range(0, k):
    #     print('Fitting model', k)
    #
    #     # Prepare the data set for fold
    #     trainData, validData, label_train, label_val = train_test_split(X_train,target.target_flag, test_size=0.4)
    #     trainDataL, trainLabels, trainIDs, trainData = prepLGB(trainData,
    #                                                            label=label_train,
    #                                                            IDCol='cl_id')
    #     validDataL, validLabels, validIDs, validData = prepLGB(validData,
    #                                                            label=label_val,
    #                                                            IDCol='cl_id')
    #     # Train
    #     gbm = lgb.train(params,
    #                     trainDataL,
    #                     100000,
    #                     valid_sets=[trainDataL, validDataL],
    #                     early_stopping_rounds=50,
    #                     verbose_eval=4)
    #
    #     # Plot importance
    #     lgb.plot_importance(gbm)
    #     plt.show()
    #
    #     # Predict
    #     predsValid += gbm.predict(validData, num_iteration=gbm.best_iteration) / k
    #     predsTrain += gbm.predict(trainData, num_iteration=gbm.best_iteration) / k
    #     predsTest += gbm.predict(X_test, num_iteration=gbm.best_iteration) / k

    # print('Start predicting...')
    # y_pred = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)
    # write_submission(y_pred, test_cl_id, 'sub/lgb.csv')

else:
    #XGB
    #preprocessing
    df = preprocessing(df)

    X_train = df.iloc[train_cl_id]
    X_test = df.iloc[test_cl_id]

    X_train.drop(columns=['cl_id'], axis=1, inplace = True)
    X_test.drop(columns=['cl_id'], axis=1, inplace = True)


    from xgboost import XGBClassifier

    def apply_model(tr, te, target):
        clf = XGBClassifier(seed=0, learning_rate=0.02, max_depth=5, subsample=0.8, colsample_bytree=0.701, n_estimators=100, nthread=8)
        clf.fit(tr, target)
        return clf.predict_proba(te)[:, 1]

    train_xg = X_train.copy()
    target_xg = target.target_flag.copy()
    scores = []
    N_SPLITS = 3
    skf = StratifiedKFold(n_splits=N_SPLITS, random_state=17, shuffle=True)
    for train_index, test_index in tqdm_notebook(skf.split(train_xg, target_xg), total=N_SPLITS):
        X_train_xg, X_test_xg = train_xg.iloc[train_index], train_xg.iloc[test_index]
        y_train_xg, y_test_xg = target_xg.iloc[train_index], target_xg.iloc[test_index]

        scores += [roc_auc_score(y_test_xg, apply_model(X_train_xg, X_test_xg, y_train_xg))]

    print (np.mean(scores))

    pred = apply_model(X_train,X_test,target.target_flag)
    # write_submission(pred, test_cl_id, 'sub/xgb.csv')