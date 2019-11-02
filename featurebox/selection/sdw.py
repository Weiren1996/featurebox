# -*- coding: utf-8 -*-

# @Time    : 2019/10/16 10:39
# @Email   : 986798607@qq.ele_ratio
# @Software: PyCharm
# @License: BSD 3-Clause

import math

import numpy as np


class SDbw(object):
    """
    score the cluster
    this part is copy from https://github.com/zhangsj1007/Machine-Learning/blob/master/S_Dbw.py
    method source:
    Halkidi, M., Batistakis, Y., & Vazirgiannis, M. (2002). Clustering validity checking methods: part II.
    ACM Sigmod Record, 31(3), 19-27.
    """

    def __init__(self, data_cl, data_cluster_ids, center_idxs=None):
        """

        Parameters
        ----------
        data_cl: np.ndarray
            each row is a variable
        center_idxs : np.ndarray
            index of cluster center
        data_cluster_ids : np.ndarray
            label of cluster
        """
        self.data = data_cl
        self.dataClusterIds = data_cluster_ids

        if center_idxs is not None:
            self.centerIdxs = center_idxs
        else:
            self.__getCenterIdxs()

        # self.center_idxs = center_idxs

        self.clusterNum = len(self.centerIdxs)

        self.stdev = self.__getStdev()

        self.clusterDensity = []

        for i in range(self.clusterNum):
            self.clusterDensity.append(self.__density(self.data[self.centerIdxs[i]], i))

    def __getCenterIdxs(self):
        """ calculate center index """

        self.centerIdxs = []

        clusterDataMp = {}
        clusterDataIdxsMp = {}

        for i in range(len(self.data)):
            entry = self.data[i]
            clusterId = self.dataClusterIds[i]
            clusterDataMp.setdefault(clusterId, []).append(entry)
            clusterDataIdxsMp.setdefault(clusterId, []).append(i)

        for clusterId in sorted(clusterDataMp.keys()):
            subData = clusterDataMp[clusterId]
            subDataIdxs = clusterDataIdxsMp[clusterId]

            m = len(subData[0])
            n = len(subData)

            meanEntry = [0.0] * m

            for entry in subData:
                meanEntry += entry

            meanEntry /= n

            minDist = float("inf")

            centerIdx = 0

            for i in range(len(subData)):
                entry = subData[i]
                idx = subDataIdxs[i]
                dist = self.__dist(entry, meanEntry)
                if minDist > dist:
                    centerIdx = idx
                    minDist = dist

            self.centerIdxs.append(centerIdx)

    def __getStdev(self):
        stdev = 0.0

        for i in range(self.clusterNum):
            varMatrix = np.var(self.data[self.dataClusterIds == i], axis=0)
            stdev += math.sqrt(np.dot(varMatrix.T, varMatrix))

        stdev = math.sqrt(stdev) / self.clusterNum

        return stdev

    def __density(self, center, cluster_idx):

        density = 0

        clusterData = self.data[self.dataClusterIds == cluster_idx]
        for i in clusterData:
            if self.__dist(i, center) <= self.stdev:
                density += 1

        return density

    def __Dens_bw(self):

        variance = 0
        if self.clusterNum == 1:
            return 0
        else:
            for i in range(self.clusterNum):
                for j in range(self.clusterNum):
                    if i == j:
                        continue
                    center = self.data[self.centerIdxs[i]] + self.data[self.centerIdxs[j]] / 2
                    interDensity = self.__density(center, i) + self.__density(center, j)
                    try:
                        variance += interDensity / max(self.clusterDensity[i], self.clusterDensity[j])
                    except ZeroDivisionError:
                        variance += 0
            return variance / (self.clusterNum * (self.clusterNum - 1))

    def __Scater(self):
        thetaC = np.var(self.data, axis=0)
        thetaCNorm = math.sqrt(np.dot(thetaC.T, thetaC))

        thetaSumNorm = 0

        for i in range(self.clusterNum):
            clusterData = self.data[self.dataClusterIds == i]
            theta_ = np.var(clusterData, axis=0)
            thetaNorm_ = math.sqrt(np.dot(theta_.T, theta_))
            thetaSumNorm += thetaNorm_

        return (1 / self.clusterNum) * (thetaSumNorm / thetaCNorm)

    @staticmethod
    def __dist(entry1, entry2):
        return np.linalg.norm(entry1 - entry2)  # 计算data entry的欧拉距离

    def result(self):
        """ return result"""
        return self.__Dens_bw() + self.__Scater()
