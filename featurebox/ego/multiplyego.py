#!/usr/bin/python
# coding:utf-8


"""
sample
"""
import warnings
from collections.abc import Iterable
from functools import partial

import joblib
import sklearn.utils
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import check_array

warnings.filterwarnings("ignore")


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


def search_space(*arg):
    meshes = np.meshgrid(*arg)
    meshes = [_.ravel() for _ in meshes]
    meshes = np.array(meshes).T
    return meshes


class MutilplyEgo:
    """
    EFO
    """

    def __init__(self, searchspace, X, y, number, regclf, feature_slice=None, n_jobs=2):
        self.n_jobs = n_jobs
        check_array(X, ensure_2d=True, force_all_finite=True)
        check_array(y, ensure_2d=True, force_all_finite=True)
        check_array(searchspace, ensure_2d=True, force_all_finite=True)
        assert X.shape[1] == searchspace.shape[1]
        self.searchspace = searchspace
        self.X = X
        self.y = y
        self.sign = []

        assert isinstance(regclf, Iterable)
        assert len(list(regclf)) >= 2
        self.regclf = list(regclf)
        self.dim = len(list(regclf))

        if feature_slice is None:
            feature_slice = tuple([tuple(range(X.shape[1]))] * self.dim)
        assert isinstance(feature_slice, (tuple, list))
        assert isinstance(feature_slice[1], (tuple, list))
        assert self.dim == len(feature_slice) == self.y.shape[1]
        self.feature_slice = feature_slice

        self.meanandstd_all = []
        self.predict_y_all = []
        self.Ei = np.zeros_like(searchspace[:, 1])
        self.Pi = np.zeros_like(searchspace[:, 1])
        self.L = np.zeros_like(searchspace[:, 1])
        self.front_point = np.zeros_like(self.y[:, 1])
        self.number = number
        self.center = np.zeros_like(searchspace[:, 1])

    def _fit(self, x, y, searchspace0, regclf0):
        def fit_parllize(random_state):
            data_train, y_train = sklearn.utils.resample(x, y, n_samples=None, replace=True,
                                                         random_state=random_state)
            regclf0.fit()
            predict_data = regclf0.predict(searchspace0)
            predict_data.ravel()
            return predict_data

        njobs = self.n_jobs

        predict_dataj = parallize(n_jobs=njobs, func=fit_parllize, iterable=range(self.number))

        return np.array(predict_dataj)

    @staticmethod
    def _mean_and_std(predict_dataj):
        mean = np.mean(predict_dataj, axis=0)
        std = np.std(predict_dataj, axis=0)
        data_predict = np.column_stack((mean, std))
        # print(data_predict.shape)
        return data_predict

    def Fit(self, regclf_number=None):

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
                predict_y = np.array(self._fit(self.X[:, feature_slicei], yi, self.searchspace[:, feature_slicei],
                                               regclfi))
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

    def pareto_front_point(self):
        sign = self.sign
        y = self.y
        m = y.shape[0]
        n = y.shape[1]
        if not sign:
            sign = np.array([1] * n)
        y = y * sign
        front_point = []
        for i in range(m):
            data_new = y[i, :].reshape(1, -1) - y
            data_max = np.max(data_new, axis=1)
            data_in = np.min(data_max)
            if data_in >= 0:
                front_point.append(i)
        self.front_point_index = front_point
        self.front_point = self.y[front_point, :].T
        return front_point

    # def CalculateEI2(self):
    #     predict_y_all = self.predict_y_all
    #     front_y = self.pareto_front_point()
    #     front_y = self.y[front_y, :].T
    #     meanstd = self.meanandstd_all
    #     pi_all = 1
    #     center = []
    #     for y_i, meanstd_i in zip(front_y, meanstd):
    #         std_ = meanstd_i[:, 1]
    #         mean_ = meanstd_i[:, 0]
    #         y_max = max(y_i)
    #         upper_bound = (mean_ - y_max) / std_
    #         numerator = [integrate.quad(lambda x: 1 / np.sqrt(2 * np.pi) * x * np.exp(-0.5 * x ** 2),
    #                                     a=upper_bound_i, b=np.inf,
    #                                     full_output=0)[0] for upper_bound_i in upper_bound]
    #         pi_i_denominator = stats.norm.cdf(upper_bound)
    #         center_i = numerator / pi_i_denominator
    #         pi_all *= pi_i_denominator
    #         center.append(center_i)
    #     center = np.array(center)
    #     self.Pi = pi_all
    #     self.center = center
    #     L = np.array([[spatial.distance.euclidean(y_j, j) for j in center.T] for y_j in front_y.T])
    #     L_min = np.min(L, axis=0)
    #     self.L = L_min

    def CalculateL(self):
        front_y = self.pareto_front_point()
        front_y = self.y[front_y, :].T
        meanstd = np.array(self.meanandstd_all)
        meanstd = meanstd[:, :, 0].T
        alll = []
        for front_y_i in front_y.T:
            l_i = np.min(front_y_i - meanstd, axis=1)
            alll.append(l_i)
        dmin = np.array(alll)
        dmaxmin = np.max(dmin, axis=0)
        self.L = dmaxmin
        return dmaxmin

    def CalculateEi(self):
        self.CalculatePi()
        self.CalculateL()
        Ei = self.L * self.Pi
        self.Ei = Ei
        return Ei

    def CalculatePi(self):
        predict_y_all = self.predict_y_all
        front_y = self.pareto_front_point()
        front_y = self.y[front_y, :].T
        tile_all = []
        for i in predict_y_all.T:
            tile = 0
            for front_y_i in front_y.T:
                big = i - front_y_i
                big_bool = np.max(big, axis=1) < 0
                tile |= big_bool
            tile_all.append(tile)
        pi = np.sum(1 - np.array(tile_all), axis=1) / self.number
        self.Pi = pi
        return pi

    def Rank(self):
        bianhao = np.arange(0, len(self.searchspace.shape[1]))
        result1 = np.column_stack((bianhao, self.Pi, self.L, self.Ei, self.center))
        max_paixu = np.argsort(result1[:, -1])
        result1 = result1[max_paixu]
        return result1


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import os

    warnings.filterwarnings("ignore")

    os.chdir(r'C:/Users/Administrator/Desktop/')
    svr = joblib.load("SVR")
    svr_el = joblib.load("svr-el")
    file_path = r'C:/Users/Administrator/Desktop/对应版.csv'
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-2].values
    y = data.iloc[:, -2:].values

    pca = PCA()
    X = pca.fit_transform(X)

    scalar = StandardScaler()
    X = scalar.fit_transform(X)

    searchspace = [
        np.arange(0, 0.6, 0.3),
        np.arange(0, 2, 1),
        np.arange(0, 2, 1),
        np.array([8, 16]),
        np.array([9, 18]),
        np.arange(0, 5, 2.5),
        np.arange(800, 1300, 250),
        np.arange(200, 800, 300),
        np.array([20, 80, 138, 250]),
    ]
    searchspace = search_space(*searchspace)
    searchspace = pca.transform(searchspace)
    searchspace = scalar.transform(searchspace)
    me = MutilplyEgo(searchspace, X, y, 100, [svr, svr_el], feature_slice=None, n_jobs=2)
    meanandstd = me.Fit()
    front_point = me.pareto_front_point()
    pi = me.CalculateEi()
