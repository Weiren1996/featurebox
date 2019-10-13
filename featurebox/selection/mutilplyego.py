#!/usr/bin/python
# coding:utf-8


"""
sample
"""
from collections import Iterable
from functools import partial
import numpy as np
from scipy import stats
import sklearn.utils
from sklearn.utils import check_array
from joblib import Parallel, delayed, effective_n_jobs


def parallize(n_jobs, func, iterable, **kwargs):
    """
    parallize the function for iterable.
    use in if __name__ == "__main__":

    Parameters
    ----------
    n_jobs:int
    cpu numbers
    func:
    function to calculate
    iterable:
    interable object
    kwargs:
    kwargs for function

    Returns
    -------
    function results
    """

    func = partial(func, **kwargs)
    if effective_n_jobs(n_jobs) == 1:
        parallel, func = list, func
    else:
        parallel = Parallel(n_jobs=n_jobs)
        func = delayed(func)

    return parallel(func(iter_i) for iter_i in iterable)


class MutilplyEgo:
    """
    EFO
    """

    def __init__(self, searchspace, data, y, regclf, feature_slice=None, n_jobs=2):
        self.n_jobs = n_jobs
        check_array(data, ensure_2d=True, force_all_finite=True)
        check_array(y, ensure_2d=True, force_all_finite=True)
        check_array(searchspace, ensure_2d=True, force_all_finite=True)
        assert data.shape[1] == data.searchspace[1]
        self.searchspace = searchspace
        self.data = data
        self.y = y

        assert isinstance(regclf, Iterable)
        assert len(list(regclf)) >= 2
        self.regclf = list(regclf)
        self.dim = len(list(regclf))

        if feature_slice is None:
            feature_slice = tuple([tuple(range(data.shape[0]))] * self.dim)
        assert isinstance(feature_slice, (tuple, list))
        assert isinstance(feature_slice[0], (tuple, list))
        assert self.dim == len(feature_slice) == self.y.shape[1]
        self.feature_slice = feature_slice

        self.meanandstd_all = []
        self.predict_y_all = []
        self.Ei = np.zeros_like(self.y[:, 1])
        self.Pi = np.zeros_like(self.y[:, 1])
        self.L = np.zeros_like(self.y[:, 1])

    @staticmethod
    def _fit(data0, y, searchspace0, number0, regclf0):

        def fit_parllize(random_state):
            data_train, y_train = sklearn.utils.resample(data0, y, n_samples=None, replace=True,
                                                         random_state=random_state)
            regclf0.fit(data_train, y_train)
            predict_data = regclf0.predict(searchspace0)
            predict_data.ravel()

        predict_dataj = parallize(n_jobs=3, func=fit_parllize, iterable=range(number0))

        return np.array(predict_dataj)

    @staticmethod
    def _mean_and_std(predict_dataj):
        mean = np.mean(predict_dataj, axis=1)
        std = np.std(predict_dataj, axis=1)
        data_predict = np.column_stack((mean, std))
        # print(data_predict.shape)
        return data_predict

    def Fit(self, number0, regclf_number=None):
        if regclf_number is None:
            contain = list(range(self.dim))
        elif isinstance(regclf_number, int):
            contain = [regclf_number]
        elif isinstance(regclf_number, (list, tuple)):
            contain = regclf_number
        else:
            raise TypeError()
        meanandstd = []
        predict_y_all = []
        for i, feature_slicei, yi, regclfi in zip(range(self.dim), self.feature_slice, self.y.T, self.regclf):
            if i in contain:
                predict_y = self._fit(self.data[:, feature_slicei], yi, self.searchspace[:, feature_slicei], number0,
                                      regclfi)
                predict_y_all.append(predict_y)

                meanandstd_i = self._mean_and_std(predict_y)
                meanandstd.append(meanandstd_i)
            else:
                pass
        predict_y_all = np.array(predict_y_all)
        if regclf_number is None:
            self.meanandstd_all = meanandstd
            self.predict_y_all = predict_y_all
        return meanandstd

    def CalculatePi(self):
        y = self.y
        meanstd = self.meanandstd_all
        pi_all = 1
        for y_i, meanstd_i in zip(y, meanstd):
            std_ = meanstd_i[:, 1]
            mean_ = meanstd_i[:, 0]
            y_min = min(y_i)
            upper_bound = (y_min - mean_) / std_
            pi_i = stats.norm.cdf(upper_bound)
            pi_all *= pi_i
        self.Pi = pi_all
        return pi_all

    def CalculateL(self):
        # todo y,self.predict_y_all
        self.Pi = []
        pass

    def CalculateEi(self):
        Ei = self.CalculateL() * self.CalculatePi()
        self.Ei = Ei
        return Ei

    def Rank(self):

        bianhao = np.arange(0, len(self.searchspace.shape[1]))
        result1 = np.column_stack((bianhao, self.Pi, self.L, self.Ei))
        max_paixu = np.argsort(result1[:, -1])
        result1 = result1[max_paixu]
        return result1
