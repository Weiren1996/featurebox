# -*- coding: utf-8 -*-

# @Time   : 2019/5/25 17:28
# @Author : Administrator
# @Project : feature_preparation
# @FileName: exhaustion.py
# @Software: PyCharm

"""
the estimator_ is the model with the best feature rather than all feature combination
"""

import warnings
from functools import partial
from itertools import combinations
import numpy as np
from joblib import effective_n_jobs, Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone
from sklearn.feature_selection.base import SelectorMixin
from sklearn.model_selection import cross_val_score
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted, check_X_y
from .mutibase import MutiBase

warnings.filterwarnings("ignore")


class Exhaustion(BaseEstimator, MetaEstimatorMixin, SelectorMixin, MutiBase):
    """
    the estimator_ is the model with the best feature rather than all feature combination
    """

    def __init__(self, estimator, n_select=(3, 4), muti_grade=2, muti_index=None, must_index=None, n_jobs=1):
        """

        :param estimator: and sklearn model or GridSearchCV
        :param n_select: the n_select list,default,n_select=(3, 4)
        :param muti_grade:muti_grade=2
        :param muti_index:the range of muti_grade
        :param must_index:the columns force to index
        :param n_jobs:default=1
        """
        super().__init__(muti_grade=muti_grade, muti_index=muti_index, must_index=must_index)
        self.estimator = estimator
        self.score_ = []
        self.n_jobs = n_jobs
        self.n_select = [n_select, ] if isinstance(n_select, int) else n_select

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def fit(self, X, y):
        """Fit the baf model and then the underlying estimator on the selected
           feature.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_feature]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """
        return self._fit(X, y)

    def _fit(self, x, y):

        def score_pri(slices, x0, y0):
            slices = list(slices)
            if len(slices) < 1:
                score0 = - np.inf
            else:
                slices = self.feature_unfold(slices)
                data_x0 = x0[:, slices]

                self.estimator._fit(data_x0, y0)
                if hasattr(self.estimator, 'best_score_'):
                    score0 = np.mean(self.estimator.best_score_)
                else:
                    score0 = np.mean(cross_val_score(self.estimator, data_x0, y0, cv=5))
            print(slices, score0)
            return score0

        score = partial(score_pri, x0=x, y0=y)

        self.score_ = []
        x, y = check_X_y(x, y, "csc")
        assert all((self.check_must, self.check_muti)) in [True, False]

        feature_list = list(range(x.shape[1]))
        fold_feature_list = self.feature_fold(feature_list)
        if self.check_must:
            fold_feature_list = [i for i in fold_feature_list if i not in self.check_must]

        slice_all = [combinations(fold_feature_list, i) for i in self.n_select]
        slice_all = [list(self.feature_must_fold(_)) for i in slice_all for _ in i]

        if effective_n_jobs(self.n_jobs) == 1:
            parallel, func = list, score
        else:
            parallel = Parallel(n_jobs=self.n_jobs)
            func = delayed(score)

        scores = parallel(func(slices) for slices in slice_all)

        feature_combination = [self.feature_unfold(_) for _ in slice_all]
        index = np.argmax(scores)
        select_feature = feature_combination[index]
        su = np.zeros(x.shape[1], dtype=np.bool)
        su[select_feature] = 1
        self.best_score_ = max(scores)
        self.score_ = scores
        self.support_ = su
        self.estimator_ = clone(self.estimator)
        self.estimator_._fit(x[:, select_feature], y)
        self.n_feature_ = len(select_feature)
        self.score_ex = list(zip(feature_combination, scores))
        self.scatter = list(zip([len(i) for i in slice_all], scores))
        self.score_ex.sort(key=lambda x: x[1], reverse=True)

        return self

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        """Reduce X to the selected feature and then predict using the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_feature]
            The input samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def score(self, X, y):
        """Reduce X to the selected feature and then return the score of the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_feature]
            The input samples.

        y : array of shape [n_samples]
            The target values.
        """
        check_is_fitted(self, 'estimator_')
        return self.estimator_.score(self.transform(X), y)

    def _get_support_mask(self):
        check_is_fitted(self, 'support_')
        return self.support_
