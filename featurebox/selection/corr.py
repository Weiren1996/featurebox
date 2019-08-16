#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/27 16:57
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
# calculate the the correction of columns,or group columns
"""
import copy
import random
import numpy as np
from .mutibase import MutiBase


class Corr(MutiBase):
    """
    fit-->shrink_cof-->count_coef-->remove_coef
    """

    def __init__(self, threshold=0.85, muti_grade=2, muti_index=None, must_index=None):
        """

        :param threshold:
        :param muti_grade:group size,calculate the correction between group
        :param muti_index:the range of muti_grade
        :param must_index:the columns force to index
        """
        super().__init__(muti_grade=muti_grade, muti_index=muti_index, must_index=must_index)
        self.threshold = threshold
        self.cov = None
        self.cov_shrink = None
        self.shrink_list = []

    def fit(self, data):

        cov = np.corrcoef(data, rowvar=False)
        cov = np.nan_to_num(cov-1)+1
        self.cov = cov
        self.shrink_list = list(range(self.cov.shape[0]))
        self._shrink_coef()

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

    def _shrink_coef(self):
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
                index.extend([i+le-1 for i in muti])
                index.sort()
                cov_muti_all.append(cov[index][:, index])
                le -= 1
            cov_muti_all = np.array(cov_muti_all)
            cov_new = np.max(cov_muti_all, axis=0)
            self.cov_shrink = cov_new
            return self.cov_shrink
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
            # com.remove(i)
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
