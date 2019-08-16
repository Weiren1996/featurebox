#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/28 16:26
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
this is a description
"""

from __future__ import division
from itertools import permutations
import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from featurebox.featurizer.basefeaturizer import BaseFeaturizer


class PolyFeaturizer(BaseFeaturizer):

    def __init__(self, degree=2):
        if isinstance(degree, (float, int)):
            degree = [degree, ]
        if 0 not in degree:
            degree.append(0)
        self.degree = degree

    @staticmethod
    def _combinations(n_features, degree):
        assert len(degree)**n_features <= 1e6, "too much degree to calculate, plese depress the degree"
        return permutations(degree, n_features)

    def fit(self, X, y=None):
        n_samples, n_features = check_array(X, accept_sparse=True).shape
        self.n_input_features_ = n_features
        return self

    def get_feature_names(self, input_features=None):

        check_is_fitted(self, 'n_input_features_')
        if input_features is None:
            input_features = ['x%d' % i for i in range(self.n_input_features_)]
        combinations = self._combinations(self.n_input_features_, self.degree)
        feature_names = []
        for rows in combinations:
            names = "*".join(["{}^{}".format(feature,row) for row,feature in zip(rows,input_features) if row != 0])

            feature_names.append(names)
        return feature_names

    def transform(self, X, y=None):
        check_is_fitted(self, ['n_input_features_'])

        X = check_array(X, dtype=FLOAT_DTYPES, accept_sparse='csc')
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        combinations = self._combinations(n_features, self.degree,)
        columns = []
        for comb in combinations:
            out_col = 1
            for col_idx, combi in enumerate(comb):
                out_col = X[:, col_idx]**combi*out_col
            columns.append(out_col)
        XP = np.vstack(columns).T
        return XP

