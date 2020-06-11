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
import copy
import functools
import itertools
import numbers
import warnings
from collections import Counter
from copy import deepcopy
from functools import partial
from operator import itemgetter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn import metrics, preprocessing
from sklearn.cluster import DBSCAN
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.utils import check_X_y, check_random_state
from sklearn.utils.multiclass import type_of_target

from featurebox.tools.tool import parallelize

warnings.filterwarnings("ignore")


def displacement(binary_distance, print_noise=0.001):
    rd = check_random_state(0)
    q = rd.random_sample(binary_distance.shape) * print_noise / 10
    binary_distance = binary_distance + q

    indexs = np.argwhere(binary_distance <= 0)
    indexs = indexs[np.where(indexs[:, 0] > indexs[:, 1])]
    t = rd.random_sample(indexs.shape[0]) * print_noise / 20
    binary_distance[indexs[:, 0], indexs[:, 1]] = t
    binary_distance[indexs[:, 1], indexs[:, 0]] = t
    return binary_distance


def cluster_printing(slices, node_color, edge_color_pen=0.7, binary_distance=None, print_noise=0.001,
                     node_name=None, highlight=None):
    """

        Parameters
        ----------
        highlight:list
            change shape
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
        node_name: list
            x_name of node
    """
    plt.figure()
    g = nx.Graph()

    def _my_ravel(data_cof):
        for i in range(data_cof.shape[0]):
            for k in range(i + 1, data_cof.shape[0]):
                yield i, k, data_cof[i, k]

    distances = displacement(binary_distance, print_noise=print_noise)

    distance_weight = list(_my_ravel(distances))
    g.add_weighted_edges_from(distance_weight)
    # edges=nx.get_edge_attributes(g, 'weight').items()
    edges, weights = zip(*nx.get_edge_attributes(g, 'weight').items())
    weights = weights / max(weights)
    pos = nx.layout.kamada_kawai_layout(g)  # calculate site

    if node_name is None:
        le = binary_distance.shape[0] or len(slices)
        lab = {i: i for i in range(le)}
    else:
        assert binary_distance.shape[0] or len(slices) == len(node_name)
        if isinstance(node_name, list) and isinstance(node_name[0], list):
            strr = ","
            node_name = [strr.join(i) for i in node_name]
        lab = {i: j for i, j in enumerate(node_name)}

    nodesize = [600] * distances.shape[0]
    node_edge_color = ["w"] * distances.shape[0]
    if highlight:
        for i in highlight:
            node_edge_color[i] = "aqua"
        for i in highlight:
            nodesize[i] *= 1.3

    mp = {-1: 'gray', 0: "mediumpurple", 1: 'seagreen', 2: 'goldenrod', 3: 'deeppink', 4: "chocolate",
          5: "lightseagreen", }
    node_color = list(map(lambda x: mp[x], node_color))

    nx.draw(g, pos, edgelist=edges, edge_color=np.around(weights, decimals=3) ** edge_color_pen,
            edge_cmap=plt.cm.Blues_r, edge_labels=nx.get_edge_attributes(g, 'weight'), edge_vmax=0.7, width=weights,
            labels=lab, font_size=12,
            node_color=np.array(node_color), vmin=-1, max=10,
            edgecolors=node_edge_color, linewidths=1,
            node_cmap=plt.cm.Paired, node_size=nodesize,
            )

    plt.show()


def score_group(cl_data, label0):
    """
    score the group results

    Parameters
    ----------
    cl_data: np.ndarray
        cluster import data_cluster
    label0: np.ndarray
        node distance matrix

    Returns
    -------
    group_score_i: list
        score for this group_i result
    """
    # sdbw = SDbw(cl_data, label0)
    # res = sdbw.result()
    if max(list(Counter(label0).items()), key=itemgetter(1))[1] > 0.8 * len(label0):
        res = -np.inf
    elif 2 <= max(label0) + 1 < len(label0):
        # res = -metrics.davies_bouldin_score(cl_data, label0)
        res = -metrics.calinski_harabasz_score(cl_data, label0)

    else:
        res = -np.inf
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
    label_0: np.ndarray, 1D
        label for node, single groups are separated
        [0,3,1,1,0,0,1,2]
    """
    db = DBSCAN(algorithm='auto', eps=epsi, metric='precomputed',
                metric_params=None, min_samples=2, n_jobs=None, p=None)
    distances /= np.max(distances)
    db.fit(distances)

    label_i = db.labels_
    set_label = list(set(label_i))
    group_i = [[i for i in range(len(label_i)) if label_i[i] == aim] for aim in set_label]

    label_0 = deepcopy(label_i)
    for j in range(len(label_0)):
        if label_0[j] == -1:
            label_0[j] = max(label_0) + 1

    return group_i, label_i, label_0


class GS(object):
    """
    grouping selection

    calculate the predict_y of base estimator on node
    calculate the distance of predict_y
    cluster the nodes by distance and get groups
    select the candidate nodes in each groups with penalty of size of node (feature number)
    rank the candidate nodes


    """

    def __init__(self, estimator, slices, estimator_i=0, n_jobs=2, batch_size=1):
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
        scorer_all = []
        cv_all = []
        for i in self.estimator:
            assert isinstance(i, GridSearchCV)
            scorer_all.append(get_scorer(i.scoring))
            cv_all.append(get_scorer(i.cv))
        self.predict_y = []  # changed with estimator_i
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        scorer = scorer_all[0]
        scorer_func = scorer._score_func
        self.metrics_method = scorer_func
        self.cv = copy.deepcopy(cv_all[0])

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
        cv = self.cv
        if isinstance(cv, numbers.Integral):
            if (type_of_target(y) in ('binary', 'multiclass')):
                cv = StratifiedKFold(cv)
            else:
                cv = KFold(cv)
        self.kf = list(cv.split(x))

    @functools.lru_cache(1024)
    def _cv_predict(self, slices_i, estimator_i):
        """ Fit y with in slices_i,resample for n_resample times """
        estimator = self.estimator[estimator_i]
        assert isinstance(estimator, GridSearchCV)
        x0 = self.x0
        y0 = self.y0
        kf = self.kf
        slices_i = list(slices_i)
        if len(slices_i) <= 1:
            raise ValueError("feature number should large than 1")
        else:
            data_x0 = x0[:, slices_i]
            estimator.fit(data_x0, y0)
            x, y = data_x0, y0

            s_estimator = copy.deepcopy(estimator.best_estimator_)

            y_test_predict_all = []
            for train, test in kf:
                X_train, X_test, y_train, y_test = x[train], x[test], y[train], y[test]
                s_estimator.fit(X_train, y_train)
                y_test_predict = s_estimator.predict(X_test)
                y_test_predict_all.append(y_test_predict)

        return y_test_predict_all

    def _cv_predict_all(self, slices=None, estimator_i=0):
        """ calculate binary distance of 2 nodes """

        self.estimator_i = estimator_i if isinstance(estimator_i, int) else self.estimator_i
        n_jobs = self.n_jobs
        batch_size = self.batch_size
        slices = slices if slices else self.slices

        ret = self.check_prop("cv_predict_all", estimator_i=self.estimator_i, slices=slices)

        if ret is not None:
            pass
        else:
            cal_score = partial(self.predict)
            ret = parallelize(n_jobs=n_jobs, func=cal_score, iterable=slices, batch_size=batch_size)

            self.add_prop("cv_predict_all", estimator_i=self.estimator_i, slices=slices, values=ret)

        self.slices = slices

        return ret

    def predict(self, slice_i):
        """change type """
        estimator_i = self.estimator_i
        return self._cv_predict(tuple(slice_i), estimator_i)

    def cv_score(self, slices_i):

        y_test_predict_all = self.predict(slices_i)
        test_index = [i[1] for i in self.kf]
        y_test_true_all = [self.y0[_] for _ in test_index]
        score = [self.metrics_method(i, j) for i, j in zip(y_test_true_all, y_test_predict_all, )]
        score_mean = np.mean(score)

        return score_mean

    def cv_score_all(self, slices=None, estimator_i=0):
        """score all node with r2

        Parameters
        ----------
        slices : list, or None, default spath.slices
            change to new slices to calculate
            the lists of the index of feature subsets, each feature subset is a node,each int is the index of X
            Examples 3 nodes
            [[1,4,5],[1,4,6],[1,2,7]]
        estimator_i: int, default spath.estimator_i
            change to the estimator_i to calculate

        Returns
        ----------
            score_mean_std: nd.ndarray 2D
            the mean and std

        """

        self.estimator_i = estimator_i if isinstance(estimator_i, int) else self.estimator_i
        n_jobs = self.n_jobs
        batch_size = self.batch_size
        slices = slices if slices else self.slices

        ret = self.check_prop("cv_score_all", estimator_i=self.estimator_i, slices=slices)

        if ret is not None:
            pass
        else:
            cal_score = partial(self.cv_score)
            ret = parallelize(n_jobs=n_jobs, func=cal_score, iterable=slices, batch_size=batch_size)

            self.add_prop("cv_score_all", estimator_i=self.estimator_i, slices=slices, values=ret)

        self.slices = slices
        return np.array(ret)

    def cal_y_distance(self, slice1):
        """ calculate binary distance of 2 nodes """
        set1 = set(slice1)

        test_index = [i[1] for i in self.kf]
        y_true_all = [self.y0[_] for _ in test_index]

        y_1_all = self.predict(set1)
        y_2_all = y_1_all

        # distance = [spatial.distance.euclidean(i, j) for i, j, k in zip(y_true_all, y_1_all, y_2_all)]
        distance = [1 - self.metrics_method(i, j) for i, j, k in zip(y_true_all, y_1_all, y_2_all)]
        distance = np.mean(distance)
        distance = distance if distance >= 0 else 0
        return distance

    def cal_y_distance_all(self, slices=None, estimator_i=0):
        """ calculate binary distance of 2 nodes """

        self.estimator_i = estimator_i if isinstance(estimator_i, int) else self.estimator_i
        n_jobs = self.n_jobs
        batch_size = self.batch_size
        slices = slices if slices else self.slices

        ret = self.check_prop("cal_y_distance_all", estimator_i=self.estimator_i, slices=slices)

        if ret is not None:
            pass
        else:
            cal_score = partial(self.cal_y_distance)
            ret = parallelize(n_jobs=n_jobs, func=cal_score, iterable=slices, batch_size=batch_size)

            self.add_prop("cal_y_distance_all", estimator_i=self.estimator_i, slices=slices, values=ret)

        self.slices = slices
        return np.array(ret)

    def cal_binary_distance(self, slice1, slice2):
        """ calculate binary distance of 2 nodes """
        set1 = set(slice1)
        set2 = set(slice2)

        test_index = [i[1] for i in self.kf]
        y_true_all = [self.y0[_] for _ in test_index]
        y_1_all = self.predict(set1)
        y_2_all = self.predict(set2)

        # distance = [spatial.distance.euclidean(j, k) for i, j, k in zip(y_true_all, y_1_all, y_2_all)]
        distance = [1 - self.metrics_method(j, k) for i, j, k in zip(y_true_all, y_1_all, y_2_all)]
        distance = np.mean(distance)
        distance = distance if distance >= 0 else 0
        return distance

    def cal_binary_distance_all(self, slices=None, estimator_i=0):
        """ calculate the distance matrix of slices """
        self.estimator_i = estimator_i if isinstance(estimator_i, int) else self.estimator_i
        n_jobs = self.n_jobs
        batch_size = self.batch_size
        slices = slices if slices else self.slices

        ret = self.check_prop("cal_binary_distance_all", estimator_i=self.estimator_i, slices=slices)

        if ret is not None:
            pass
        else:
            cal_binary_distance = partial(self.cal_binary_distance)
            slices_cuple = list(itertools.product(slices, repeat=2))
            ret = parallelize(n_jobs=n_jobs, func=cal_binary_distance, iterable=slices_cuple, respective=True,
                              batch_size=batch_size)
            ret = np.reshape(ret, (len(slices), len(slices)), order='F')

            self.add_prop("cal_binary_distance_all", estimator_i=self.estimator_i, slices=slices, values=ret)

        self.slices = slices
        return ret

    def check_prop(self, prop, estimator_i=0, slices=None):
        if slices == self.slices:
            if hasattr(self, "".join(["result", prop, "_", "%s" % estimator_i])):
                return getattr(self, "".join(["result", prop, "_", "%s" % estimator_i]), None)

    def add_prop(self, prop, slices, estimator_i, values):
        if hasattr(self, "".join(["result", prop, "_", "%s" % estimator_i])) and slices == self.slices:
            pass
        else:
            setattr(self, "".join(["result", prop, "_", "%s" % estimator_i]), values)

    # @time_this_function
    def cal_group(self, eps=None, printing=False, slices=None, estimator_i=0,
                  print_noise=0.1, node_name=None, pre_binary_distance_all=None):
        """

        Parameters
        ----------

        eps: int
            args for DBSCAN
        printing: bool
            default,True for GS and False for UGS
        slices : list, or None, default spath.slices
            change to new slices to calculate
            the lists of the index of feature subsets, each feature subset is a node,each int is the index of X
            Examples 3 nodes
            [[1,4,5],[1,4,6],[1,2,7]]
        estimator_i: int, default spath.estimator_i
            change to the estimator_i to calculate
        print_noise: int
            add noise for less printing overlap
        node_name: list of str
            x_name of node, be valid for printing is True
        pre_binary_distance_all
            pre-calculate binary_distance_all ,please make sure the slices and estimator_i are same
            as the slices and estimator_i in pre-calculate binary_distance_all

        Returns
        -------
        group: list
            group results, the result of groups is unique
            Examples 4 groups for 8 node
            [[0,4,5],[2,3,6],[7],[1]]
        """
        self.estimator_i = estimator_i if isinstance(estimator_i, int) else self.estimator_i
        self.slices = slices if slices else self.slices
        slices = self.slices

        if isinstance(print_noise, (float, int)) and 0 < print_noise <= 1:
            pass
        else:
            raise TypeError("print_noise should be in (0,1]")

        if not isinstance(pre_binary_distance_all, np.ndarray):
            binary_distance = self.cal_binary_distance_all(slices, self.estimator_i)
        else:
            binary_distance = pre_binary_distance_all

        pre_y = self._cv_predict_all(slices, self.estimator_i)
        pre_y = np.array([np.concatenate([i.ravel() for i in pre_yi]).T for pre_yi in pre_y])

        distances = binary_distance
        if eps:
            group, label, label_sep = sc(eps, distances)
        else:
            group = None
            group_score = -np.inf
            label = [[1] * len(self.slices)]

            for epsi in np.arange(0.05, 0.95, 0.01):
                group_i, label_i, label_sep_i = sc(epsi, distances)
                group_score_i = score_group(pre_y, label_sep_i)
                print(label_i, group_score_i)
                if group_score_i > group_score:
                    group_score = group_score_i
                    # print(epsi, group_score)
                    group = group_i
                    label = label_i

        if printing:
            self.cluster_print(binary_distance, print_noise=print_noise, node_name=node_name, label=label)
        group_last = group.pop()
        group += [[i] for i in group_last]
        self.group = group
        return group

    def cluster_print(self, binary_distance, label=None, eps=0.1, print_noise=0.001, node_name=None, highlight=None):
        """ print temporary"""
        if label is None:
            group, label, label_sep = sc(eps, binary_distance)
        else:
            label = label
        cluster_printing(slices=self.slices, binary_distance=binary_distance,
                         print_noise=print_noise, node_name=node_name,
                         node_color=label, highlight=highlight)

    # def distance_print(spath):

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

        score_all = np.array(self.cv_score_all(self.slices))

        score = score_all[:, 0]
        std = score_all[:, 1]
        max_std = np.max(std)
        t = 2
        m = np.array([len(i) for i in self.slices])

        score = score * (1 - std / max_std) - alpha * (np.exp(m - t) + 1)
        score = preprocessing.minmax_scale(score)

        score_select, selected, site_select = self._select(self.slices, self.group, score, fliters=False, )
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

        result = sorted(zip(score_select, selected, site_select), key=itemgetter(0), reverse=True)
        result = list(zip(*result))
        return result


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

    def __init__(self, estimator, slices, estimator_n=None, n_jobs=2, estimator_i=0, batch_size=1):
        """

        Parameters
        ----------
        estimator : list
            list of base estimator or GridSearchCV from sklearn
        slices: list
            the lists of the index of feature subsets, each feature subset is a node,each int is the index of X
            Examples 3 nodes
            [[1,4,5],[1,4,6],[1,2,7]]
        estimator_n: list
            default indexes of estimator
        """
        super().__init__(estimator, slices, estimator_i, n_jobs=n_jobs, batch_size=batch_size)
        if estimator_n is None:
            self.estimator_n = list(range(len(estimator)))
        else:
            self.estimator_n = estimator_n
        assert len(self.estimator) >= 2
        assert len(self.estimator_n) >= 2

    def cal_t_group(self, eps=None, printing=False, pre_group=None):
        """

        Parameters
        ----------
        eps: int
            args for DBSCAN
        printing: bool
            draw or not, default, False
        pre_group: None or list of different estimator's groups
            the sort of pre_group should match to spath.estimator !
            pre-calculate results by spath.cal-group for all base estimator, to escape double counting


        Returns
        -------
        tn_group: list
            the element of tournament groups can be repeated
            Examples
            [[1,2,3],[3,4,5],[1,6,7],[2,3]]

        """

        slices = [tuple(_) for _ in self.slices]
        if not pre_group:
            group_result = [self.cal_group(estimator_i=i, eps=eps, printing=printing, print_noise=0.01, node_name=None)
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

        score_all = np.array([self.cv_score_all(estimator_i=i)[:, 0] for i in self.estimator_n])

        score = np.mean(np.array(score_all), axis=0)
        std = np.std(np.array(score_all), axis=0)
        max_std = np.max(std)
        t = 2
        m = np.array([len(i) for i in self.slices])

        score = score * (1 - std / max_std) - alpha * (np.exp(m - t) + 1)
        score = preprocessing.minmax_scale(score)
        score_select, selected, site_select = self._select(self.slices, self.group, score, fliters=True)

        result = sorted(zip(score_select, selected, site_select), key=itemgetter(0), reverse=True)
        result = list(zip(*result))
        return result
