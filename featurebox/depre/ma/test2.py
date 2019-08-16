#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/28 14:15
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
this is a description
"""

import numpy as np
import pandas as pd
import functools
import warnings
from copy import deepcopy
import sympy
from scipy import optimize
from scipy.special._ufuncs import erf
from sklearn import utils
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import assert_all_finite, check_array

from featurebox.selection.score import score_muti
from featurebox.tools.exports import Store
from featurebox.tools.imports import Call

store = Store(r'C:\Users\Administrator\Desktop\ma')
data = Call(r'C:\Users\Administrator\Desktop\ma', index_col=None)
data_import = data.csv.ma
data_import = data_import.iloc[np.where(330 <= data_import['t'])[0]]
data_import = data_import.iloc[np.where(1280 >= data_import['t'])[0]]
x = data_import[["k", "t"]]
data_import[["k"]] = data_import[["k"]] * 10 ** 15
y = data_import["c"].values

# scal = preprocessing.MinMaxScaler()
# x = scal.fit_transform(x)
x, y = utils.shuffle(x, y, random_state=5)
x = x.values

poly = PolynomialFeatures(list(np.arange(0.1, 2, 0.2)))
x_tran = poly.fit_transform(x)
Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001,
      warm_start=False, positive=False, random_state=None)
res = score_muti(x_tran, y, me="reg", paras=True, method_name=["Lasso-L1"], shrink=1, str_name=False,
                 param_grid={"alpha": [50]})
coef = res[1].sparse_coef_.data
intercept = res[1].intercept_
names = np.array(poly.get_feature_names())[res[1].sparse_coef_.indices]
score = res[0]
expr = "+".join(["({}*{})".format(*i) for i in zip(coef, names)]) + "+(%s)" % intercept
