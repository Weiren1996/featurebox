# -*- coding: utf-8 -*-

# @Time    : 2019/11/6 16:20
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

import numpy as np
from sklearn.utils._random import check_random_state

'''
    """please copy the code to test dont use by import"""

    X = init(n_feature=10, m_sample=100)  # 样本数

    X_name = ["X%s" % i for i in range(X.shape[1])]
    for i, X_i in enumerate(X_name):
        locals()[X_i] = X[:, i]

    # 添加关系
    """relation"""

    X2 = X0+X1

    """noise"""

    X0 = add_noise(X0,ratio=0.01)


    X0 = add_noise(X0, ratio=0.2)
    X3 = add_noise(X0 / X2,ratio=0.2)
    X4 = add_noise(X0 * X2,ratio=0.2)

    """重定义"""
    X_all = [eval("X%s" % i) for i in range(X.shape[1])]
    X_new = np.vstack(X_all).T

    """定义函数"""  # 改变函数
    y = X0 ** 3 + X2
    return X, y
'''


def init(m_sample=100, n_feature=10):
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
