import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Normalizer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import hstack, vstack
from sklearn.linear_model import LogisticRegression

from tools import base_preprocassing, dict_vect_preprocessing, tfidf_preprocessing


PATH = './data/'

#read data
df = pd.read_csv(os.path.join(PATH, 'train_sample.csv'), nrows=400000, skiprows=range(1,200000))
df = df.join(pd.read_csv(os.path.join(PATH,'mlboot_train_answers.tsv'), delimiter='\t').set_index('cuid'), on='cuid', how='inner')

#preprocess
df = dict_vect_preprocessing(df)
df = df.convert_objects(convert_numeric=True)

#try
X_train, X_test, y_train, y_test = train_test_split(df.drop(axis=1, columns=['cuid','target']),df.target, test_size=0.2, random_state=17)

n = Normalizer()
X_train = n.fit_transform(X_train)
X_test = n.transform(X_test)

lr = LogisticRegression(penalty='l2', C=1)
sf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
cv = cross_val_score(lr,X_train, y_train, scoring='roc_auc', cv=sf, n_jobs=-1)
print (cv.mean(), cv.std())
lr.fit(X_train,y_train)
pred = lr.predict_proba(X_test)
print (roc_auc_score(y_test,pred[:, 1]))
