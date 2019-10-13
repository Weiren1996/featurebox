# -*- coding: utf-8 -*-

# @Time   : 2019/7/13 19:27
# @Author : Administrator
# @Project : feature_toolbox
# @FileName: mmgs.py
# @Software: PyCharm

"""
UGS
this is a union select method for feature subsets.
key:
1.gather performance of different model
2.raise the best one from similar learning performance subsets
3.rank the raised subsets and penalty the size of subsets

node == feature subset
"""

import itertools
import warnings
from functools import partial
import matplotlib.pyplot as plt
from joblib import effective_n_jobs, Parallel, delayed
from sklearn import metrics, preprocessing
from sklearn.cluster import DBSCAN
from sklearn.utils import check_X_y, resample
import numpy as np
import networkx as nx
import math

warnings.filterwarnings("ignore")


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
            label of cluster
        data_cluster_ids : np.ndarray
            index of cluster center
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

        for i in range(self.clusterNum):
            for j in range(self.clusterNum):
                if i == j:
                    continue
                center = self.data[self.centerIdxs[i]] + self.data[self.centerIdxs[j]] / 2
                interDensity = self.__density(center, i) + self.__density(center, j)
                variance += interDensity / max(self.clusterDensity[i], self.clusterDensity[j])

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


def cluster_printing(slices, node_color, edge_color_pen=0.7, binary_distance=None, print_noise=1,
                     node_name=None):
    """

        Parameters
        ----------
        slices: list
            the lists of the index of feature subsets, each feature subset is a node.
            Examples 3 nodes
            [[1,4,5],[1,4,6],[1,2,7]]
        node_color: np.ndarray 1D, list, the same size as slices
            the label to classify the node
        edge_color_pen: int
            the transparency of edge between node
        binary_distance: np.ndarray
            distance matrix for each pair node
        print_noise: int
            add noise for less printing overlap
        node_name: list of str
            name of node
    """
    from numpy import random
    g = nx.Graph()

    def _my_ravel(data_cof):
        for i in range(data_cof.shape[0]):
            for k in range(i + 1, data_cof.shape[0]):
                yield i, k, data_cof[i, k]

    random.seed(0)
    q = random.random(binary_distance.shape) * print_noise / 4
    binary_distance = binary_distance - q

    indexs = np.argwhere(binary_distance <= 0)
    indexs = indexs[np.where(indexs[:, 0] > indexs[:, 1])]
    t = random.random(indexs.shape[0]) * print_noise / 5
    binary_distance[indexs[:, 0], indexs[:, 1]] = t
    binary_distance[indexs[:, 1], indexs[:, 0]] = t

    distances = binary_distance

    nodesize = [600] * distances.shape[0]
    distance_weight = list(_my_ravel(distances))
    g.add_weighted_edges_from(distance_weight)
    # edges=nx.get_edge_attributes(g, 'weight').items()
    edges, weights = zip(*nx.get_edge_attributes(g, 'weight').items())

    pos = nx.layout.kamada_kawai_layout(g)  # calculate site

    if node_name is None:
        le = binary_distance.shape[0] or len(slices)
        lab = {i: i for i in range(le)}
    else:
        assert binary_distance.shape[0] or len(slices) == len(node_name)
        if isinstance(node_name[0], list):
            strr = ","
            node_name = [strr.join(i) for i in node_name]
        lab = {i: j for i, j in enumerate(node_name)}
    nx.draw(g, pos, edgelist=edges, edge_color=np.around(weights, decimals=3) ** edge_color_pen, labels=lab,
            font_size=12,
            edge_cmap=plt.cm.Blues_r, edge_labels=nx.get_edge_attributes(g, 'weight'), edge_vmax=0.7,
            node_color=np.array(node_color) + 1, vmin=-1, max=10,
            node_cmap=plt.cm.Paired, node_size=nodesize, width=weights,
            )
    plt.show()


def score_group(cl_data, label):
    """
    score the group results

    Parameters
    ----------
    cl_data: np.ndarray
        cluster import data_cluster
    label: np.ndarray
        node distance matrix

    Returns
    -------
    group_score_i: list
        score for this group_i result
    """
    sdbw = SDbw(cl_data, label)
    res = sdbw.result()
    return res


def sc(epsi, distances):
    """
    cluster the node and get group results

    Parameters
    ----------
    epsi: int
        args for DBSCAN
    distances: np.ndarray
        distance matrix

    Returns
    -------
    group_i: list
        Examples 4 groups for 8 node
        [[0,4,5],[2,3,6],[7],[1]]
    label_i: np.ndarray, 1D
        label for node, the same label is the same group
        Examples 4 groups for 8 node
        [0,3,1,1,0,0,1,2]
    """
    db = DBSCAN(algorithm='auto', eps=epsi, metric='precomputed',
                metric_params=None, min_samples=2, n_jobs=None, p=None)
    db.fit(distances)

    label_i = db.labels_
    set_label = list(set(label_i))
    group_i = [[i for i in range(len(label_i)) if label_i[i] == aim] for aim in set_label]
    return group_i, label_i


class GS(object):
    """
    grouping selection

    calculate the predict_y of base estimator on node
    calculate the distance of predict_y
    cluster the nodes by distance and get groups
    select the candidate nodes in each groups with penalty of size of node (feature number)
    rank the candidate nodes


    """

    def __init__(self, estimator, slices, estimator_i=0):
        """

        Parameters
        ----------
        estimator : list
            list of base estimator or GridSearchCV from sklearn
        slices: list
            the lists of the index of feature subsets, each feature subset is a node,each int is the index of X
            Examples 3 nodes
            [[1,4,5],[1,4,6],[1,2,7]]
        estimator_i: int
            default index of estimator
        """
        self.slices = slices
        self.estimator_i = estimator_i
        if isinstance(estimator, list):
            self.estimator = estimator
        else:
            self.estimator = [estimator, ]
        self.predict_y = []  # changed with estimator_i

    def fit(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
        y: np.ndarray, 1D
        """
        x, y = check_X_y(x, y, "csc")
        self.x0 = x
        self.y0 = y

    def predict(self, slices_i):
        """ predict y with in slices_i,resample for n_resample times """
        n_resample = 500
        estimator = self.estimator[self.estimator_i]
        x0 = self.x0
        y0 = self.y0
        slices_i = list(slices_i)
        if len(slices_i) <= 1:
            y_predict_all = np.zeros((x0.shape[0], n_resample))
        else:
            data_x0 = x0[:, slices_i]
            estimator.fit(data_x0, y0)
            if hasattr(estimator, 'best_estimator_'):
                estimator = estimator.best_estimator_
            else:
                pass
            # y_predict = cross_val_predict(estimator, data_x0, y0, cv=5)

            y_predict_all = []
            for i in range(n_resample):
                data_train, y_train = resample(data_x0, y0, n_samples=None, replace=True, random_state=i)
                estimator.fit(data_train, y_train)
                y_predict = estimator.predict(data_x0)
                y_predict_all.append(y_predict)
            y_predict_all = np.array(y_predict_all)

        return y_predict_all

    def score(self, slices_i):
        """
        Parameters
        ----------
        slices_i:list
            the index of X.

        Returns
        -------
            r2 score mean
            r2 score std

        """
        y_pre = self.predict(list(slices_i))

        score = [metrics.r2_score(y_pre_i, self.y0) for y_pre_i in y_pre.T]
        score = [score_i if score_i >= 0 else 0 for score_i in score]
        score_mean = np.mean(score)
        score_std = np.mean(score)
        return score_mean, score_std

    def score_all(self, slices=None, n_jobs=1, estimator_i=0):
        """score all node with r2

        Parameters
        ----------
        slices : list, or None, default self.slices
            change to new slices to calculate
            the lists of the index of feature subsets, each feature subset is a node,each int is the index of X
            Examples 3 nodes
            [[1,4,5],[1,4,6],[1,2,7]]
        estimator_i: int, default self.estimator_i
            change to the estimator_i to calculate
        n_jobs: int

        Returns
        ----------
            score_mean_std: nd.ndarray 2D
            the mean and std

        """

        self.estimator_i = estimator_i
        self.predict_y = []
        if slices is None:
            slices = self.slices
        else:
            self.slices = slices

        cal_score = partial(self.score)

        if effective_n_jobs(n_jobs) == 1:
            parallel, func = list, cal_score
        else:
            parallel = Parallel(n_jobs=n_jobs)
            func = delayed(cal_score)

        score_mean_std = parallel(func(slicesi) for slicesi in slices)
        return score_mean_std

    def predict_mean(self, slices_i):
        """ calculate the mean of all predict_y """
        y_predict_all = self.predict(slices_i)
        y_mean = np.mean(y_predict_all, axis=1)

        return y_mean

    def cal_binary_distance(self, slice1, slice2):
        """ calculate binary distance of 2 nodes """
        set1 = set(slice1)
        set2 = set(slice2)
        set0 = set1 & set2
        y1 = self.predict_mean(set1)
        y2 = self.predict_mean(set2)
        y0 = self.predict_mean(set0)

        distance = 1 - metrics.r2_score(y1 - y0, y2 - y0)
        distance = distance if distance >= 0 else 0
        self.predict_y.append(y1)
        return distance

    def cal_binary_distance_all(self, slices=None, n_jobs=1):
        """ calculate the distance matrix of slices """
        if slices is None:
            slices = self.slices
        else:
            self.slices = slices

        cal_binary_distance = partial(self.cal_binary_distance)
        if effective_n_jobs(n_jobs) == 1:
            parallel, func = list, cal_binary_distance
        else:
            parallel = Parallel(n_jobs=n_jobs)
            func = delayed(cal_binary_distance)
        slices_cuple = itertools.product(slices, repeat=2)
        distance = parallel(func(*slicesi) for slicesi in slices_cuple)
        distance = np.reshape(distance, (len(slices), len(slices)), order='F')
        return distance

    def cal_group(self, eps=0.1, printing=False, pre_binary_distance_all=None, slices=None, estimator_i=0,
                  print_noise=1, node_name=None):
        """

        Parameters
        ----------
        eps: int
            args for DBSCAN
        printing: bool
            default,True for GS and False for UGS
        pre_binary_distance_all:
            pre-calculate results by self.binary_distance_all, to escape double counting
        slices : list, or None, default self.slices
            change to new slices to calculate
            the lists of the index of feature subsets, each feature subset is a node,each int is the index of X
            Examples 3 nodes
            [[1,4,5],[1,4,6],[1,2,7]]
        estimator_i: int, default self.estimator_i
            change to the estimator_i to calculate
        print_noise: int
            add noise for less printing overlap
        node_name: list of str
            name of node, be valid for printing is True

        Returns
        -------
        group: list
            group results, the result of groups is unique
            Examples 4 groups for 8 node
            [[0,4,5],[2,3,6],[7],[1]]
        """
        self.estimator_i = estimator_i
        self.predict_y = []

        if slices is None:
            slices = self.slices
        else:
            self.slices = slices

        if isinstance(print_noise, (float, int)) and 0 < print_noise <= 1:
            pass
        else:
            raise TypeError("print_noise should be in (0,1]")

        if not isinstance(pre_binary_distance_all, np.ndarray):
            binary_distance = self.cal_binary_distance_all(slices, n_jobs=3)
        else:
            binary_distance = pre_binary_distance_all
        distances = binary_distance
        pre_y = self.predict_y
        if eps:
            group, label = sc(eps, distances)
            # group_score_i = score_group(pre_y, label)
        else:
            group = None
            group_score = 0
            label = [[1] * len(self.slices)]

            for epsi in np.arange(0.05, 0.95, 0.05):
                group_i, label_i = sc(epsi, distances)

                group_score_i = score_group(pre_y, label_i)
                if group_score_i > group_score:
                    group = group_i
                    label = label_i

        if printing:
            cluster_printing(slices=slices, binary_distance=binary_distance,
                             print_noise=print_noise, node_name=node_name,
                             node_color=label)
        self.group = group
        return group

    def select_gs(self, alpha=0.01):
        """

        Parameters
        ----------
        alpha: int
            penalty coefficient

        Returns
        -------
        score_select: list
            selected node score
        selected: list
            selected node in import
        site_select: list
            selected node number in import
        """
        slices = self.slices

        score = self.score_all(slices, n_jobs=3)

        score = score[0, :]
        score = np.mean(np.array(score), axis=0)
        std = np.std(np.array(score), axis=0)
        max_std = np.max(std)
        t = 2
        m = np.array([len(i) for i in self.slices])
        score = score * (1 - std / max_std) - alpha * (np.exp(m - t) + 1)
        score = preprocessing.minmax_scale(score)

        score_select, selected, site_select = self._select(slices, self.group, score, fliters=False, )
        return score_select, selected, site_select

    @staticmethod
    def _select(slices, group, score, fliters=False):
        """select the maximum, greater is better. Not suit for minimum"""
        score_groups = [[score[i] for i in slicei_group] for slicei_group in group]
        select = [np.argmax(i) for i in score_groups]  # select in group_i
        site_select = [i[_] for _, i in zip(select, group)]  # site
        site_select = list(set(site_select)) if fliters else site_select
        score_select = [score[_] for _ in site_select]  # score_select
        selected = [slices[_] for _ in site_select]  # select
        return score_select, selected, site_select


class UGS(GS):
    """
    union grouping selection

    calculate the predict_y  on node, for each base estimator
    calculate the distance of predict_y, for each base estimator
    cluster the nodes by distance and get groups, for each base estimator
    unite groups of base estimators to tournament groups
    select the candidate nodes in each groups with penalty of size of node (feature number)
    rank the candidate nodes


    """

    def __init__(self, estimator, slices, estimator_i=0):
        """

        Parameters
        ----------
        estimator : list
            list of base estimator or GridSearchCV from sklearn
        slices: list
            the lists of the index of feature subsets, each feature subset is a node,each int is the index of X
            Examples 3 nodes
            [[1,4,5],[1,4,6],[1,2,7]]
        estimator_i: int
            default index of estimator
        """
        super().__init__(estimator, slices, estimator_i)
        self.estimator_n = list(range(len(estimator)))
        assert len(estimator) >= 2

    def cal_t_group(self, eps=0.1, printing=False, pre_group=None):
        """

        Parameters
        ----------
        eps: int
            args for DBSCAN
        printing: bool
            draw or not, default, False
        pre_group: None or list of different estimator's groups
            the sort of pre_group should match to self.estimator !
            pre-calculate results by self.cal-group for all base estimator, to escape double counting


        Returns
        -------
        tn_group: list
            the element of tournament groups can be repeated
            Examples
            [[1,2,3],[3,4,5],[1,6,7],[2,3]]

        """

        slices = [tuple(_) for _ in self.slices]
        if not pre_group:
            group_result = [self.cal_group(estimator_i=i, eps=eps, printing=printing, print_noise=1, node_name=None)
                            for i in self.estimator_n]
        else:
            assert len(pre_group) == self.estimator_n, "the size of pre_group should is the number fo estimator!"
            group_result = pre_group
        for group in group_result:
            single = group.pop()
            single = [[_] for _ in single]
            group.extend(single)
        tn_group = []
        for slicei in range(len(slices)):
            slicei_group = set()
            for group in group_result:
                for groupi in group:
                    if slicei in groupi:
                        slicei_group.update(groupi)
            slicei_group = list(slicei_group)
            tn_group.append(slicei_group)
        # todo hebing ?
        self.group = tn_group
        self.group_result = []
        self.group_result.append(group_result)
        return tn_group

    def select_ugs(self, alpha=0.01):
        """

        Parameters
        ----------
        alpha: int
            penalty coefficient

        Returns
        -------
        score_select: list
            selected node score
        selected: list
            selected node in import
        site_select: list
            selected node number in import
        """
        score = np.array([self.score_all(n_jobs=3, estimator_i=i) for i in self.estimator_n])
        score = score[0, :, :]
        score = np.mean(np.array(score), axis=0)
        std = np.std(np.array(score), axis=0)
        max_std = np.max(std)
        t = 2
        m = np.array([len(i) for i in self.slices])
        score = score * (1 - std / max_std) - alpha * (np.exp(m - t) + 1)
        score = preprocessing.minmax_scale(score)
        score_select, selected, site_select = self._select(self.slices, self.group, score, fliters=True)
        return score_select, selected, site_select


data = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 2], [2, 2, 2], [2, 2, 3]])
data_cluster = np.array([1, 0, 1, 2, 2])
centers_index = np.array([1, 0, 3])
a = SDbw(data, data_cluster, centers_index)
# a = S_Dbw(data_cluster, data_cluster)
print(a.result())

# if __name__ == '__main__':
# import warnings
# import pandas as pd
# from sklearn import preprocessing, utils
# import numpy as np
#
# warnings.filterwarnings("ignore")
# store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp_last')
# Data = Call(r'C:\Users\Administrator\Desktop\band_gap_exp_last')
# all_import_structure = Data.csv.all_import_structure
# data_import = all_import_structure.drop(
#     ["name", "structure", "structure_type", "space_group", "reference", 'material_id', 'composition'], axis=1)
#
# data216_import = data_import.iloc[np.where(data_import['group_number'] == 216)[0]]
# data225_import = data_import.iloc[np.where(data_import['group_number'] == 225)[0]]
# data186_import = data_import.iloc[np.where(data_import['group_number'] == 186)[0]].drop("BeO186", axis=0)
# data216_225_import = pd.concat((data216_import, data225_import))
# data_cluster = data225_import.values
#
# X = data_cluster[:, 2:]
# y = data_cluster[:, 0]
# #
# scal = preprocessing.MinMaxScaler()
# X = scal.fit_transform(X)
# X, y = utils.shuffle(X, y, random_state=5)
# index = Data.pickle_pd.t2019testGS
# index2 = Data.pickle_pd.t2019testGSGPR
#
# me1 = SVR(kernel='rbf', gamma='auto', degree=3, tol=1e-3, epsilon=0.01, shrinking=True, max_iter=3000)
# cv1 = 5
# scoring1 = 'r2'
# param_grid1 = [{'C': [1.0e7, 1.0e5, 1.0e3, 10, 1, 0.1]}]
#
# kernel = 1 * Matern(nu=2.5)
# param_grid6 = [{'alpha': [1e-12, 1e-10, 1e-9, 1e-7, 1e-5, 1e-3, 1e-3, 1e-1],
#                 'kernel': kernel}]
# me6 = gaussian_process.GaussianProcessRegressor(kernel=kernel,
#                                                 alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0,
#                                                 normalize_y=False, copy_X_train=True, random_state=0)
# cv6 = 5
# scoring6 = 'r2'
#
# estimator = GridSearchCV(me1, cv=cv1, param_grid=param_grid1, scoring=scoring1, n_jobs=1)
# # estimator2 = GridSearchCV(me6, cv=cv6, param_grid=param_grid6, scoring=scoring6, n_jobs=1)
#
# gs = GS([estimator])
# gs._fit(X, y)
# y = gs.predict(index[0][0])
# y_score = gs.score(index[0][0])
# y_banary = gs.cal_binary_distance(index[6][0], index[5][0], estimator_i=0)
#
# index = [tuple(i) for i in list(zip(*index))[0][:20]]
# # y_banary_all = gs.cal_binary_distance_all(index, estimator_i=0)
# cal_group = gs.cal_group(index, eps=0.2, printing=True)
# select_gs = gs.select_gs(index, cal_group, estimator_i=0, theshold=0.001, greater_is_better=True)
# # index = [tuple(i) for i in list(zip(*index))[0][:10]]
# # index2 = [tuple(i) for i in list(zip(*index2))[0][:10]]
# # gs = MMGS([me1, me6])
# # gs._fit(X, y)
# # index_all = list(set(index) | set(index2))
# # slice_group = gs.group_rank(index_all)
