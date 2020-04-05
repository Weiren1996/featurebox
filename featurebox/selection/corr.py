#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/27 16:57
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
# calculate the the correction of columns,or cal_group columns
"""
import copy
import random

import numpy as np
from sklearn.datasets import load_boston

from featurebox.selection.mutibase import MutiBase
from featurebox.tools.tool import name_to_name


class Corr(MutiBase):
    """
    _fit-->shrink_cof-->count_coef-->remove_coef
    """

    def __init__(self, threshold=0.85, muti_grade=2, muti_index=None, must_index=None):
        """

        :param threshold:
        :param muti_grade:cal_group size,calculate the correction between cal_group
        :param muti_index:the range of muti_grade
        :param must_index:the columns force to index
        """
        super().__init__(muti_grade=muti_grade, muti_index=muti_index, must_index=must_index)
        self.threshold = threshold
        self.cov = None
        self.cov_shrink = None
        self.shrink_list = []

    @staticmethod
    def MIC_matirx(dataframe,alpha=0.6, c=15):

        from minepy import MINE
        mine = MINE(alpha=alpha, c=c)
        data = np.array(dataframe)
        n = len(data[0, :])
        result = np.zeros([n, n])

        for i in range(n):
            for j in range(n):
                mine.compute_score(data[:, i], data[:, j])
                result[i, j] = mine.mic()
                result[j, i] = mine.mic()
        RT = result
        return RT

    def fit(self, data, pre_cal=None, method="mean",alpha=0.6, c=15):
        if pre_cal is None:

            cov = np.corrcoef(data, rowvar=False,)
        elif pre_cal is "mic":
            cov = self.MIC_matirx(data, alpha=alpha, c=c)

        elif isinstance(pre_cal, np.ndarray) and pre_cal.shape[0] == data.shape[1]:
            cov = pre_cal
        else:
            raise TypeError("pre_cal is None or coef of data_cluster with shape(data_cluster[0],data_cluster[0])")
        cov = np.nan_to_num(cov - 1) + 1
        self.cov = cov
        self.data = data
        self.shrink_list = list(range(self.cov.shape[0]))
        self._shrink_coef(method=method)

    def _shrink_coef(self, method="mean" or "max"):

        if self.check_muti:
            self.shrink_list = list(range(self.cov.shape[0]))
            self.shrink_list = list(self.feature_fold(self.shrink_list))

            cov = self.cov
            single = tuple([i for i in self.shrink_list if i not in self.check_muti])
            muti = tuple([i for i in self.shrink_list if i in self.check_muti])

            cov_muti_all = []
            le = self.muti_grade
            while le:
                index = []
                index.extend(single)
                index.extend([i + le - 1 for i in muti])
                index.sort()
                cov_muti_all.append(cov[index][:, index])
                le -= 1
            cov_muti_all = np.array(cov_muti_all)
            if method is "mean":
                cov_new = np.mean(cov_muti_all, axis=0)
            else:
                cov_new = np.max(cov_muti_all, axis=0)
            self.cov_shrink = cov_new
            return self.cov_shrink
        else:
            pass

    def transform_index(self, data):
        if isinstance(data, int):
            return self.shrink_list.index(data)
        elif isinstance(data, (list, tuple)):
            return [self.shrink_list.index(i) for i in data]

    def inverse_transform_index(self, data):
        if isinstance(data, int):
            return self.shrink_list[data]
        elif isinstance(data, (list, tuple)):
            return [self.shrink_list[i] for i in data]
        else:
            pass

    def transform(self, data):
        if isinstance(data, (list, tuple)):
            return data[self.shrink_list]
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            return data[self.shrink_list]
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            return data[:, self.shrink_list]
        else:
            pass

    def count_cof(self, cof=None):

        """check cof and count the number"""
        if cof is None:
            cof = self.cov_shrink
        if cof is None:
            raise NotImplemented("imported cof is None")

        list_count = []
        list_coef = []
        g = np.where(abs(cof) >= self.threshold)
        for i in range(cof.shape[0]):
            e = np.where(g[0] == i)
            com = list(g[1][e])
            # ele_ratio.remove(i)
            list_count.append(com)
            list_coef.append([cof[i, j] for j in com])
        self.list_coef = list_coef
        self.list_count = list_count
        return list_coef, list_count

    @staticmethod
    def remove_coef(cof_list_all):
        """delete the index of feature with repeat coef """
        random.seed(0)
        reserve = []
        for i in cof_list_all:
            if not cof_list_all:
                reserve.append(i)

        for cof_list in cof_list_all:
            if not cof_list:
                pass
            else:
                if reserve:
                    candi = []
                    for j in cof_list:

                        con = any([[False, True][j in cof_list_all[k]] for k in reserve])
                        if not con:
                            candi.append(j)
                    if any(candi):
                        a = random.choice(candi)
                        reserve.append(a)
                    else:
                        pass
                else:
                    a = random.choice(cof_list)
                    reserve.append(a)
                cof_list_t = copy.deepcopy(cof_list)
                for dela in cof_list_t:
                    for cof_list2 in cof_list_all:
                        if dela in cof_list2:
                            cof_list2.remove(dela)
        return reserve

    @staticmethod
    def cov_y(x_, y_):
        cov = np.corrcoef(x_, y_, rowvar=False)
        cov = cov[:, -1][:-1]
        return cov

    def remove_by_y(self, y_):
        corr = self.cov_y(self.data, y_)
        corr = self.feature_fold(self, corr)
        lcount = self.list_count
        fea_all = []
        score = name_to_name(corr, search=lcount, search_which=0, return_which=(1,), two_layer=True)
        for score_i, list_i in zip(score, lcount):
            indexs = np.argmax(score_i)
            feature_index = list_i[indexs]
            fea_all.append(feature_index)

        fea_all = sorted(list(set(fea_all)))
        return fea_all


if __name__ == "__main__":
    x, y = load_boston(return_X_y=True)
    co = Corr(threshold=0.7)
    c = co.count_cof(np.corrcoef(x, rowvar=False))[1]


    # pandas as pd
    # import matplotlib.pyplot as plt
    # import numpy as np
    #

    #
    # x = np.linspace(0, 1, 1000)
    # y = np.sin(10 * np.pi * x) + x
    # mine = MINE(alpha=0.6, c=15)
    # mine.compute_score(x, y)
    #
    # print("Without noise:")
    # print("MIC", mine.mic())
    # print()
    #
    # np.random.seed(0)
    # y += np.random.uniform(-1, 1, x.shape[0])  # add some noise
    # mine.compute_score(x, y)
    #
    #
    # def MIC_matirx(dataframe, mine):
    #
    #     data = np.array(dataframe)
    #     n = len(data[0, :])
    #     result = np.zeros([n, n])
    #
    #     for i in range(n):
    #         for j in range(n):
    #             mine.compute_score(data[:, i], data[:, j])
    #             result[i, j] = mine.mic()
    #             result[j, i] = mine.mic()
    #     RT = result
    #     return RT
    #
    #
    # mine = MINE(alpha=0.6, c=5)
    # data_wine_mic = MIC_matirx(
    #     np.array([[1, 2, 3, 4, 5, 1, 5, 2, 1, 4], [2, 3, 4, 1, 2, 1, 3, 1, 6, 1], [6, 4, 5, 3, 4, 1, 4, 1, 6, 2]]).T,
    #     mine)

