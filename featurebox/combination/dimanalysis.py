# -*- coding: utf-8 -*-

# @Time    : 2019/12/2 19:21
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import numpy as np
from numpy.linalg import matrix_rank
from sklearn.utils import check_array

from featurebox.combination.dim import Dim


def dimension_check(x, y=None):
    if y is not None:
        x.append(y)
    x = np.array(x).T
    x = check_array(x, ensure_2d=True)
    x = x.astype(np.float64)
    det = matrix_rank(x)
    che = []
    for i in range(x.shape[1]):
        x_new = np.delete(x, i, 1)
        det2 = matrix_rank(x_new)
        che.append(det - det2)
    sum(che)

    if sum(che) == 0:
        return 1
    else:
        return 0
