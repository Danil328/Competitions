import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from numpy.random import normal


def simple_blending(basic_algorithms, meta_algorithm, X_train, X_test, y_train, part1_ratio=0.9, random_state=None):
    X_train_part1, X_train_part2,\
    y_train_part1, y_train_part2 = train_test_split(X_train, y_train, test_size=1-part1_ratio, random_state=random_state)

    meta_features_part2 = np.zeros((X_train_part2.shape[0], len(basic_algorithms)))
    meta_features_test = np.zeros((X_test.shape[0], len(basic_algorithms)))

    for index, basic_algorithm in enumerate(basic_algorithms):
        basic_algorithm.fit(X_train_part1, y_train_part1)

        part2_predictions = basic_algorithm.predict_proba(X_train_part2)[:, 1]
        meta_features_part2[:, index] = part2_predictions

        test_predictions = basic_algorithm.predict_proba(X_test)[:, 1]
        meta_features_test[:, index] = test_predictions

    meta_algorithm.fit(meta_features_part2, y_train_part2)

    return meta_algorithm.predict_proba(meta_features_test)[:, 1]


def average_blending(basic_algorithms, meta_algorithm, X_train, X_test, y_train, part1_ratio=0.9, n_iter=3, random_state=None):

    simple_blending_realizations = list()
    for iter in range(n_iter):
        if random_state is None:
            realization_random_state = None
        else:
            realization_random_state = iter + random_state

        simple_blending_realizations.append(simple_blending(basic_algorithms,
                                                            meta_algorithm,
                                                            X_train, X_test, y_train,
                                                            part1_ratio=part1_ratio,
                                                            random_state=realization_random_state))
    return np.mean(np.asarray(simple_blending_realizations), axis=0)


def concatenate_blending(basic_algorithms, meta_algorithm, X_train, X_test, y_train, part1_ratio=0.9, n_iter=3, random_state=None):

    def blending_realization(basic_algorithms, X_train, y_train, X_test, part1_ratio, random_state):
        X_train_part1, X_train_part2, \
        y_train_part1, y_train_part2 = train_test_split(X_train, y_train, test_size=1 - part1_ratio,
                                                        random_state=random_state)

        meta_features_part2 = np.zeros((X_train_part2.shape[0], len(basic_algorithms)))
        meta_features_test = np.zeros((X_test.shape[0], len(basic_algorithms)))

        for index, basic_algorithm in enumerate(basic_algorithms):
            basic_algorithm.fit(X_train_part1, y_train_part1)

            part2_predictions = basic_algorithm.predict_proba(X_train_part2)[:, 1]
            meta_features_part2[:, index] = part2_predictions

            test_predictions = basic_algorithm.predict_proba(X_test)[:, 1]
            meta_features_test[:, index] = test_predictions

        return (meta_features_part2, y_train_part2, meta_features_test)

    realizations = list()
    for iter in range(n_iter):
        if random_state is None:
            realization_random_state = None
        else:
            realization_random_state = iter + random_state

        realizations.append(blending_realization(basic_algorithms, X_train, y_train, X_test, part1_ratio, random_state=realization_random_state))

    X_meta = np.concatenate([x[0] for x in realizations])
    y_meta = np.concatenate([x[1] for x in realizations])
    X_meta_test = np.concatenate([x[2] for x in realizations])

    meta_algorithm.fit(X_meta, y_meta)
    predictions = meta_algorithm.predict_proba(X_meta_test)[:, 1]

    return np.mean(predictions.reshape(-1, X_test.shape[0]), axis=0)


def classical_stacking(basic_algorithms, meta_algorithm, X_train, X_test, y_train, n_folds=3, n_iter=3, random_state=None):

    realizations = list()

    for iter in range(n_iter):
        if random_state is None:
            realization_random_state = None
        else:
            realization_random_state = iter + random_state

        folds_results = list()
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=realization_random_state)
        for train_index, test_index in skf.split(X_train, y_train):
            X_train_folds, X_test_fold = X_train[train_index], X_train[test_index]
            y_train_folds, y_test_fold = y_train[train_index], y_train[test_index]

            fold_meta_features = np.zeros((X_test_fold.shape[0], len(basic_algorithms)))

            for index, basic_algorithm in enumerate(basic_algorithms):
                basic_algorithm.fit(X_train_folds, y_train_folds)
                test_fold_predictions = basic_algorithm.predict_proba(X_test_fold)[:, 1]
                fold_meta_features[:, index] = test_fold_predictions

            folds_results.append((fold_meta_features, y_test_fold))

        meta_features = np.concatenate([x[0] for x in folds_results])
        meta_y = np.concatenate([x[1] for x in folds_results])


        meta_features_test = np.zeros((X_test.shape[0], len(basic_algorithms)))
        for index, basic_algorithm in enumerate(basic_algorithms):
            basic_algorithm.fit(X_train, y_train)
            test_predictions = basic_algorithm.predict_proba(X_test)[:, 1]
            meta_features_test[:, index] = test_predictions

        meta_algorithm.fit(meta_features, meta_y)

        realizations.append(meta_algorithm.predict_proba(meta_features_test)[:, 1])

    return np.mean(np.asarray(realizations), axis=0)


def noise_classical_stacking(basic_algorithms, meta_algorithm, X_train, X_test, y_train, n_folds=3, n_iter=3, noise_scale=0.03, random_state=None):

    realizations = list()

    for iter in range(n_iter):
        if random_state is None:
            realization_random_state = None
        else:
            realization_random_state = iter + random_state

        folds_results = list()
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=realization_random_state)
        for train_index, test_index in skf.split(X_train, y_train):
            X_train_folds, X_test_fold = X_train[train_index], X_train[test_index]
            y_train_folds, y_test_fold = y_train[train_index], y_train[test_index]

            fold_meta_features = np.zeros((X_test_fold.shape[0], len(basic_algorithms)))

            for index, basic_algorithm in enumerate(basic_algorithms):
                basic_algorithm.fit(X_train_folds, y_train_folds)
                test_fold_predictions = basic_algorithm.predict_proba(X_test_fold)[:, 1]
                fold_meta_features[:, index] = normal(test_fold_predictions, scale=noise_scale)

            folds_results.append((fold_meta_features, y_test_fold))

        meta_features = np.concatenate([x[0] for x in folds_results])
        meta_y = np.concatenate([x[1] for x in folds_results])


        meta_features_test = np.zeros((X_test.shape[0], len(basic_algorithms)))
        for index, basic_algorithm in enumerate(basic_algorithms):
            basic_algorithm.fit(X_train, y_train)
            test_predictions = basic_algorithm.predict_proba(X_test)[:, 1]
            meta_features_test[:, index] = normal(test_predictions, scale=noise_scale)

        meta_algorithm.fit(meta_features, meta_y)

        realizations.append(meta_algorithm.predict_proba(meta_features_test)[:, 1])

    return np.mean(np.asarray(realizations), axis=0)
