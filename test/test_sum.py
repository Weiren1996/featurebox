# -*- coding: utf-8 -*-
import warnings
from itertools import combinations, chain

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.utils._random import check_random_state

from featurebox.selection.exhaustion import Exhaustion
from featurebox.selection.sum import SUM
from featurebox.tools.quickmethod import dict_method_reg


def init(n_feature=10, m_sample=100):
    n = n_feature
    m = m_sample
    mean = [1] * n
    cov = np.zeros((n, n))
    for i in range(cov.shape[1]):
        cov[i, i] = 1
    rdd = check_random_state(1)
    X = rdd.multivariate_normal(mean, cov, m)
    return X


def add_noise(s, ratio):
    print(s.shape)
    rdd = check_random_state(1)
    return s + rdd.random_sample(s.shape) * np.max(s) * ratio


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    X = init(n_feature=10, m_sample=100)  # 样本数

    X_name = ["X%s" % i for i in range(X.shape[1])]
    for i, X_i in enumerate(X_name):
        locals()[X_i] = X[:, i]

    # 添加关系
    """relation"""
    '''
    X2 = X0+X1
    '''
    """noise"""
    '''
    X0 = add_noise(X0,ratio=0.01)
    '''

    X0 = add_noise(X0, ratio=0.2)
    X3 = add_noise(X0 / X2, ratio=0.2)
    X4 = add_noise(X0 * X2, ratio=0.2)

    """重定义"""
    X_all = [eval("X%s" % i) for i in range(X.shape[1])]
    X_new = np.vstack(X_all).T

    """定义函数"""  # 改变函数
    y = X0 ** 3 + X2

    """定义方法"""
    method_all = ['KNR-set', 'SVR-set', "KR-set"]
    feature_number = [2]

    print(dict_method_reg().keys())
    estimator = []
    for method in method_all:
        me2, cv2, scoring2, param_grid2 = dict_method_reg()[method]
        scoring2 = 'r2'
        gd2 = GridSearchCV(me2, cv=cv2, param_grid=param_grid2, scoring=scoring2, n_jobs=1)
        estimator.append(gd2)

    feature_list = list(range(X_new.shape[1]))
    slice_all = list(chain(*[list(combinations(feature_list, i)) for i in feature_number]))[:10]

    pre_select = True

    if pre_select:
        scores = []
        slice_all = []
        for estimatori in estimator:
            clf = Exhaustion(estimatori, n_select=feature_number, muti_grade=2, muti_index=None, must_index=None,
                             n_jobs=3)
            clf.fit(X_new, y)
            slice_all.extend(list(zip(*clf.score_ex[:10]))[0])
            scores.append(clf.score_ex[:10])
            print(clf.score_ex[0])
        slice_all = [tuple(_) for _ in slice_all]
        slice_all = list(set(slice_all))

    reg = SUM(estimator, slice_all, estimator_n=None, n_jobs=3)
    reg.fit(X_new, y)

    ss = reg.cv_score_all(estimator_i=0)
    ss0 = reg.cal_binary_distance_all(estimator_i=0)
    ss1 = reg.cal_binary_distance_all(estimator_i=1)
    ss2 = reg.cal_binary_distance_all(estimator_i=2)
    reg.cluster_print(ss0)
    reg.cluster_print(ss1)
    reg.cluster_print(ss2)

    # reg.cal_group()
    # reg.cal_t_group()
    #
    d = reg.mean_max_method()
    c = reg.distance_method()
    a = reg.kk_distance_method()
    b = reg.y_distance_method()
    e = reg.pareto_method()
    f = reg.y_distance_method()

    # result = pd.DataFrame(np.array(result))
    # result.to_csv('Result.csv')

    # result = reg.y_distance_method()
    # result = pd.DataFrame(np.array(result))
