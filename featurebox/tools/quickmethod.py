#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/27 16:57
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
# score
"""

import warnings

from sklearn import utils, kernel_ridge, gaussian_process, ensemble, linear_model, neighbors, preprocessing
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.linear_model import LogisticRegression, BayesianRidge, SGDRegressor, Lasso, ElasticNet, Perceptron
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_validate
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")


def dict_method_clf():
    dict_method = {}
    # 1st part
    """1SVC"""
    me1 = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated',
              coef0=0.0, shrinking=True, probability=False,
              tol=1e-3, cache_size=200, class_weight='balanced',
              verbose=False, max_iter=-1, decision_function_shape='ovr',
              random_state=None)
    cv1 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring1 = 'accuracy'

    param_grid1 = [{'C': [1000, 100, 10000, 10, 1], 'gamma': [0.001, 0.01, 0.0001]}]

    dict_method.update({'SVC-set': [me1, cv1, scoring1, param_grid1]})

    """2LogRL2"""
    me2 = LogisticRegression(penalty='l2', solver='liblinear', dual=False, tol=1e-3, C=1.0, fit_intercept=True,
                             intercept_scaling=1, class_weight='balanced', random_state=0)
    cv2 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring2 = 'accuracy'

    param_grid2 = [{'C': [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2]}, ]

    dict_method.update({"LogRL2-set": [me2, cv2, scoring2, param_grid2]})

    """3SGDCL2"""
    me3 = linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15,
                                     fit_intercept=True, max_iter=None, tol=None, shuffle=True,
                                     verbose=0, epsilon=0.1, random_state=0,
                                     learning_rate='optimal', eta0=0.0, power_t=0.5,
                                     class_weight="balanced", warm_start=False, average=False, n_iter=None)
    cv3 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring3 = 'accuracy'

    param_grid3 = [{'alpha': [0.0001, 0.001, 0.01]}, ]

    dict_method.update({"SGDCL2-set": [me3, cv3, scoring3, param_grid3]})

    """4KNC"""
    me4 = neighbors.KNeighborsClassifier(n_neighbors=5)
    cv4 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring4 = 'balanced_accuracy'

    param_grid4 = [{'n_neighbors': [3, 4, 5]}, ]

    dict_method.update({"KNC-set": [me4, cv4, scoring4, param_grid4]})

    """5GPC"""
    kernel = 1.0 * RBF(1.0)
    me5 = gaussian_process.GaussianProcessClassifier(kernel=kernel)
    cv5 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring5 = 'balanced_accuracy'
    param_grid5 = [{'max_iter_predict': [100, ]}, ]

    dict_method.update({'GPC-set': [me5, cv5, scoring5, param_grid5]})

    # 2nd part
    '''TreeC'''
    me6 = DecisionTreeClassifier(
        criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None, class_weight="balanced", presort=False)
    cv6 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring6 = 'accuracy'
    param_grid6 = [{'max_depth': [3, 4, 5, 6]}]
    dict_method.update({'TreeC-em': [me6, cv6, scoring6, param_grid6]})

    '''GBC'''
    me7 = ensemble.GradientBoostingClassifier(
        loss='deviance', learning_rate=0.1, n_estimators=100,
        subsample=1.0, criterion='friedman_mse', min_samples_split=2,
        min_samples_leaf=1, min_weight_fraction_leaf=0.,
        max_depth=3, min_impurity_decrease=0.,
        min_impurity_split=None, init=None,
        random_state=None, max_features=None, verbose=0,
        max_leaf_nodes=None, warm_start=False,
        presort='auto')
    cv7 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring7 = 'balanced_accuracy'
    param_grid7 = [{'max_depth': [3, 4, 5, 6]}]
    dict_method.update({'GBC-em': [me7, cv7, scoring7, param_grid7]})

    '''RFC'''
    me8 = ensemble.RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None,
                                          min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.,
                                          max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.,
                                          min_impurity_split=None, bootstrap=True, oob_score=False,
                                          random_state=None, verbose=0, warm_start=False,
                                          class_weight="balanced")
    cv8 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring8 = 'accuracy'
    param_grid8 = [{'max_depth': [3, 4, 5, 6]}]
    dict_method.update({"RFC-em": [me8, cv8, scoring8, param_grid8]})

    "AdaBC"
    me9 = AdaBoostClassifier(n_estimators=100, learning_rate=1., algorithm='SAMME.R',
                             random_state=0)
    cv9 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring9 = 'accuracy'
    param_grid9 = [{'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2),
                                       DecisionTreeClassifier(max_depth=3)]}]
    dict_method.update({"AdaBC-em": [me9, cv9, scoring9, param_grid9]})

    # 3nd

    'SGDCL1'
    me12 = linear_model.SGDClassifier(loss='hinge', penalty='l1', alpha=0.0001, l1_ratio=0.15,
                                      fit_intercept=True, max_iter=None, tol=None, shuffle=True,
                                      verbose=0, epsilon=0.1, random_state=0,
                                      learning_rate='optimal', eta0=0.0, power_t=0.5,
                                      class_weight="balanced", warm_start=False, average=False, n_iter=None)
    cv12 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring12 = 'accuracy'
    param_grid12 = [{'alpha': [0.0001, 0.001, 0.01]}, ]
    dict_method.update({"SGDC-L1": [me12, cv12, scoring12, param_grid12]})

    "Per"
    me14 = Perceptron(penalty="l1", alpha=0.0001, fit_intercept=True, max_iter=None, tol=None,
                      shuffle=True, verbose=0, eta0=1.0, random_state=0,
                      class_weight=None, warm_start=False, n_iter=None)
    cv14 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring14 = 'accuracy'
    param_grid14 = [{'alpha': [0.0001, 0.001, 0.01]}, ]
    dict_method.update({"Per-L1": [me14, cv14, scoring14, param_grid14]})

    """LogRL1"""
    me15 = LogisticRegression(penalty='l1', solver='liblinear', dual=False, tol=1e-3, C=1.0, fit_intercept=True,
                              intercept_scaling=1, class_weight='balanced', random_state=0)
    cv15 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring15 = 'accuracy'
    param_grid15 = [{'C': [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2]}, ]
    dict_method.update({"LogR-L1": [me15, cv15, scoring15, param_grid15]})

    return dict_method


def dict_method_reg():
    dict_method = {}
    # 1st part
    """1SVR"""
    me1 = SVR(kernel='rbf', gamma='auto', degree=3, tol=1e-3, epsilon=0.1, shrinking=False, max_iter=2000)
    cv1 = 5
    scoring1 = 'explained_variance'
    param_grid1 = [{'C': [1.0e8, 1.0e7, 1.0e6, 1.0e5, 10000, 1000, 100, 10, 1], 'gamma': [0.02, "auto"]}]
    dict_method.update({"SVR-set": [me1, cv1, scoring1, param_grid1]})

    """2BayesianRidge"""
    me2 = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,
                        copy_X=True, fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06,
                        n_iter=300, normalize=False, tol=0.01, verbose=False)
    cv2 = 5
    scoring2 = 'explained_variance'
    param_grid2 = [{'alpha_1': [1e-07, 1e-05, 1e-05], 'alpha_2': [1e-07, 1e-05, 1e-03]}]
    dict_method.update({'BayR-set': [me2, cv2, scoring2, param_grid2]})

    """3SGDRL2"""
    me3 = SGDRegressor(alpha=0.0001, average=False,
                       epsilon=0.1, eta0=0.01, fit_intercept=True, l1_ratio=0.15,
                       learning_rate='invscaling', loss='squared_loss', max_iter=1000,
                       penalty='l2', power_t=0.25,
                       random_state=0, shuffle=True, tol=0.01,
                       verbose=0, warm_start=False)
    cv3 = 5
    scoring3 = 'explained_variance'
    param_grid3 = [{'alpha': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 1e-05]}]
    dict_method.update({'SGDRL2-set': [me3, cv3, scoring3, param_grid3]})

    """4KNR"""
    me4 = neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                        metric='minkowski')
    cv4 = 5
    scoring4 = 'explained_variance'
    param_grid4 = [{'n_neighbors': [3, 4, 5, 6]}]
    dict_method.update({"KNR-set": [me4, cv4, scoring4, param_grid4]})

    """5kernelridge"""
    kernel = 1.0 * RBF(1.0)
    me5 = kernel_ridge.KernelRidge(alpha=1, kernel=kernel, gamma="scale", degree=3, coef0=1, kernel_params=None)
    cv5 = 5
    scoring5 = 'explained_variance'
    param_grid5 = [{'alpha': [100, 10, 1, 0.1, 0.01, 0.001]}]
    dict_method.update({'KR-set': [me5, cv5, scoring5, param_grid5]})

    """6GPR"""
    # kernel = 1.0 * RBF(1.0)
    kernel = 2 * Matern(nu=1.5)
    me6 = gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=1e-10, optimizer='fmin_l_bfgs_b',
                                                    n_restarts_optimizer=0,
                                                    normalize_y=False, copy_X_train=True, random_state=0)
    cv6 = 5
    scoring6 = 'explained_variance'
    param_grid6 = [{'alpha': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]}]
    dict_method.update({"GPR-set": [me6, cv6, scoring6, param_grid6]})

    # 2nd part

    """6RFR"""
    me7 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                         min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                         min_impurity_split=None, bootstrap=True, oob_score=False,
                                         random_state=None, verbose=0, warm_start=False)
    cv7 = 5
    scoring7 = 'explained_variance'
    param_grid7 = [{'max_depth': [3, 4, 5, 6]}]
    dict_method.update({"RFR-em": [me7, cv7, scoring7, param_grid7]})

    """7GBR"""
    me8 = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100,
                                             subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                                             min_samples_leaf=1, min_weight_fraction_leaf=0.,
                                             max_depth=3, min_impurity_decrease=0.,
                                             min_impurity_split=None, init=None, random_state=None,
                                             max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                                             warm_start=False, presort='auto')
    cv8 = 5
    scoring8 = 'explained_variance'
    param_grid8 = [{'max_depth': [3, 4, 5, 6]}]
    dict_method.update({'GBR-em': [me8, cv8, scoring8, param_grid8]})

    "AdaBR"
    me9 = AdaBoostRegressor(n_estimators=100, learning_rate=1.,
                            random_state=0)
    cv9 = 5
    scoring9 = 'explained_variance'
    param_grid9 = [{'n_estimators': [50, 100, 200]}]
    dict_method.update({"AdaBR-em": [me9, cv9, scoring9, param_grid9]})

    '''TreeR'''
    me10 = DecisionTreeRegressor(
        criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_features=None, random_state=0, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None, presort=False)
    cv10 = 5
    scoring10 = 'explained_variance'
    param_grid10 = [{'max_depth': [3, 4, 5, 6]}]
    dict_method.update({'TreeC-em': [me10, cv10, scoring10, param_grid10]})

    'ElasticNet'
    me11 = ElasticNet(alpha=1.0, l1_ratio=0.7, fit_intercept=True, normalize=False, precompute=False, max_iter=1000,
                      copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None)

    cv11 = 5
    scoring11 = 'explained_variance'
    param_grid11 = [{'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'l1_ratio': [0.3, 0.5, 0.8]}]
    dict_method.update({"ElasticNet-L1": [me11, cv11, scoring11, param_grid11]})

    'Lasso'
    me12 = Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000,
                 tol=0.001,
                 warm_start=False, positive=False, random_state=None, )

    cv12 = 5
    scoring12 = 'explained_variance'
    param_grid12 = [{'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 1000]}, ]
    dict_method.update({"Lasso-L1": [me12, cv12, scoring12, param_grid12]})

    """SGDRL1"""
    me13 = SGDRegressor(alpha=0.0001, average=False,
                        epsilon=0.1, eta0=0.01, fit_intercept=True, l1_ratio=0.15,
                        learning_rate='invscaling', loss='squared_loss', max_iter=1000,
                        penalty='l1', power_t=0.25,
                        random_state=0, shuffle=True, tol=0.01,
                        verbose=0, warm_start=False)
    cv13 = 5
    scoring13 = 'explained_variance'
    param_grid13 = [{'alpha': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7]}]
    dict_method.update({'SGDR-L1': [me13, cv13, scoring13, param_grid13]})

    return dict_method


def dict_me(me="clf"):
    if me == "clf":
        dict_method_ = dict_method_clf()
    else:
        dict_method_ = dict_method_reg()
    return dict_method_


def score_muti(x_select, y, me="reg", paras=True, method_name=None, shrink=2, str_name=False, param_grid=None):
    """score with different method
    :param param_grid: user's param_grid
    :param str_name:
    :param x_select: X
    :param y: y
    :param me: clf or reg
    :param paras: Gridsearch or not
    :param method_name: list or one name of method
    :param shrink: scale or not
    :return:
    """
    dict_method = dict_me(me=me)

    if method_name is not None:
        if isinstance(method_name, str):
            method_name = [method_name]
        dict_method = {_: dict_method[_] for _ in method_name}
        print(dict_method)

    x_select, y = utils.shuffle(x_select, y, random_state=1)
    x_select2 = preprocessing.scale(x_select)

    if len(dict_method) > 1 and param_grid is not None:
        raise IndexError("only single method can accept param_grid, please set one method_name or param_grid=None ")

    score_all = []
    estimator = []

    for method in list(dict_method.keys()):
        me2, cv2, scoring2, param_grid2 = dict_method[method]
        if paras is None:
            if me == "clf":
                scoring2 = 'balanced_accuracy'
            if me == "reg":
                scoring2 = 'r2'
            cv2 = 3
            if shrink == 1:
                score1 = cross_val_score(me2, x_select, y, scoring=scoring2, cv=cv2, n_jobs=1, verbose=0,
                                         fit_params=None).mean()
                score_all.append(score1)
            elif shrink == 2:
                score2 = cross_val_score(me2, x_select2, y, scoring=scoring2, cv=cv2, n_jobs=1, verbose=0,
                                         fit_params=None).mean()
                score_all.append(score2)
            else:
                score1 = cross_val_score(me2, x_select, y, scoring=scoring2, cv=cv2, n_jobs=1, verbose=0,
                                         fit_params=None).mean()
                score2 = cross_val_score(me2, x_select2, y, scoring=scoring2, cv=cv2, n_jobs=1, verbose=0,
                                         fit_params=None).mean()
                score3 = max(score1, score2)
                score_all.append(score3)

            scores = cross_validate(me2, x_select2, y, scoring=scoring2, cv=cv2, return_train_score=False,
                                    return_estimator=True)
            me2 = scores['estimator'][0]
            estimator.append(me2)
        else:
            if isinstance(param_grid, (dict, list)):
                param_grid2 = param_grid
            if me == "clf":
                scoring2 = 'balanced_accuracy'
            if me == "reg":
                scoring2 = 'r2'
            cv2 = 3
            if shrink == 1:
                gd2 = GridSearchCV(me2, cv=cv2, param_grid=param_grid2, scoring=scoring2, n_jobs=1)
                gd2.fit(x_select, y)
                score1 = gd2.best_score_
                score_all.append(score1)
                estimator.append(gd2.best_estimator_)
            elif shrink == 2:
                gd2 = GridSearchCV(me2, cv=cv2, param_grid=param_grid2, scoring=scoring2, n_jobs=1)
                gd2.fit(x_select2, y)
                score2 = gd2.best_score_
                score_all.append(score2)
                estimator.append(gd2.best_estimator_)
            else:
                gd2 = GridSearchCV(me2, cv=cv2, param_grid=param_grid2, scoring=scoring2, n_jobs=1)

                gd2.fit(x_select, y)
                score1 = gd2.best_score_
                gd2.fit(x_select2, y)
                score2 = gd2.best_score_
                score_all.append(max(score1, score2))
                estimator.append(gd2.best_estimator_)

    if str_name is True:
        estimator = [str(estimator[i]).split("(")[0] for i in range(len(estimator))]

    if len(score_all) == 1:
        score_all = score_all[0]
        estimator = estimator[0]
    return score_all, estimator


def method_pack(method_all, me="reg", gd=True):
    if not method_all:
        method_all = ['KNR-set', 'SVR-set', "KR-set"]
    dict_method = dict_me(me=me)

    print(dict_method.keys())
    if gd:
        estimator = []
        for method in method_all:
            me2, cv2, scoring2, param_grid2 = dict_method[method]
            if me == "clf":
                scoring2 = 'balanced_accuracy'
            if me == "reg":
                scoring2 = 'r2'
            gd2 = GridSearchCV(me2, cv=cv2, param_grid=param_grid2, scoring=scoring2, n_jobs=1)
            estimator.append(gd2)
        return estimator
    else:
        estimator = []
        for method in method_all:
            me2, cv2, scoring2, param_grid2 = dict_method[method]
            if me == "clf":
                scoring2 = 'balanced_accuracy'
            if me == "reg":
                scoring2 = 'r2'
            gd2 = cross_val_score(me2, cv=cv2, scoring=scoring2)
            estimator.append(gd2)
        return estimator


if __name__ == "__main__":
    a = dict_method_clf()
    b = dict_method_reg()
