import pandas as pd
import numpy as np
import seaborn as sns; sns.set_style('darkgrid')
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons

RANDOM_STATE = 17

X, y = make_moons(n_samples=1000, random_state=RANDOM_STATE, noise=0.4)
plt.scatter(X[:, 0], X[:, 1], c=pd.Series(y).map({0:'red', 1:'blue'}))
#plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)



from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


classifiers = [KNeighborsClassifier(metric='manhattan', n_neighbors=46),
               KNeighborsClassifier(metric='euclidean', n_neighbors=44),
               LogisticRegression(C=4, penalty='l2', class_weight='balanced', random_state=RANDOM_STATE),
               LogisticRegression(C=36, penalty='l1', class_weight='balanced', random_state=RANDOM_STATE),
               GradientBoostingClassifier(n_estimators=400, learning_rate=0.01, max_depth=2, subsample=0.3, random_state=RANDOM_STATE),
               GradientBoostingClassifier(n_estimators=400, learning_rate=0.01, max_depth=14, subsample=0.5, random_state=RANDOM_STATE)]

# for classifier in classifiers:
#     classifier.fit(X_train, y_train)
#     predictions = classifier.predict_proba(X_test)[:, 1]
#     print(roc_auc_score(y_test, predictions))
#
#
# from methods import simple_blending
# simple_blending_predictions = simple_blending(classifiers,
#                                               LogisticRegression(random_state=RANDOM_STATE),
#                                               X_train, X_test, y_train,
#                                               part1_ratio=0.9,
#                                               random_state=RANDOM_STATE)
# print(roc_auc_score(y_test, simple_blending_predictions))
#
#
# from methods import average_blending
# average_blending_predictions = average_blending(classifiers,
#                                                 LogisticRegression(random_state=RANDOM_STATE),
#                                                 X_train, X_test, y_train,
#                                                 part1_ratio=0.9,
#                                                 n_iter=10,
#                                                 random_state=RANDOM_STATE)
# print(roc_auc_score(y_test, average_blending_predictions))
# 
#
# from methods import concatenate_blending
# concatenate_blending_predictions = concatenate_blending(classifiers,
#                                                         LogisticRegression(random_state=RANDOM_STATE),
#                                                         X_train, X_test, y_train,
#                                                         part1_ratio=0.9,
#                                                         n_iter=10,
#                                                         random_state=RANDOM_STATE)
# print(roc_auc_score(y_test, concatenate_blending_predictions))


# from methods import classical_stacking
# classical_stacking_predictions = classical_stacking(classifiers,
#                                                     LogisticRegression(random_state=RANDOM_STATE),
#                                                     X_train, X_test, y_train,
#                                                     n_folds=5,
#                                                     n_iter=5,
#                                                     random_state=RANDOM_STATE)
# print(roc_auc_score(y_test, classical_stacking_predictions))



# from methods import noise_classical_stacking
# classical_stacking_predictions = noise_classical_stacking(classifiers,
#                                                     LogisticRegression(random_state=RANDOM_STATE),
#                                                     X_train, X_test, y_train,
#                                                     n_folds=5,
#                                                     n_iter=5,
#                                                     noise_scale=0.08,
#                                                     random_state=RANDOM_STATE)
# print(roc_auc_score(y_test, classical_stacking_predictions))





#0.9210970839260313
















