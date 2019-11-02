# -*- coding: utf-8 -*-

# @Time    : 2019/11/3 6:01
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import numpy as np


def init(n=4):
    mean = [1] * n
    cov = np.zeros((n, n))
    for i in range(cov.shape[1]):
        cov[i, i] = 1
    X = np.random.multivariate_normal(mean, cov, 100)
    return X


X = init(4)

y = X[:, 1] ** 3 * X[:, 2]

x1 = X[:, 1] * X[:, 2]
x2 = X[:, 1] + X[:, 2]

X[:, 0] = x1
X[:, 3] = x2
