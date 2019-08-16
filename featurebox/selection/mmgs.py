# -*- coding: utf-8 -*-

# @Time   : 2019/7/13 19:27
# @Author : Administrator
# @Project : feature_toolbox
# @FileName: mmgs.py
# @Software: PyCharm

import itertools
import warnings
from functools import partial
import matplotlib.pyplot as plt
from joblib import effective_n_jobs, Parallel, delayed
from sklearn import metrics, gaussian_process
from sklearn.cluster import DBSCAN
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.svm import SVR
from sklearn.utils import check_X_y
import numpy as np
from featurebox.tools.exports import Store
from featurebox.tools.imports import Call

warnings.filterwarnings("ignore")

"""
this is a description
"""


class GS(object):
    """
    group select

    """
    def __init__(self, estimator):
        if isinstance(estimator, list):
            self.estimator = estimator
        else:
            self.estimator = [estimator, ]

    def fit(self, x, y):

        x, y = check_X_y(x, y, "csc")
        self.x0 = x
        self.y0 = y

    def predict(self, slices, estimator_i=0):
        """
        predict y with in slices.

        Parameters
        ----------
        slices:list
        the index of X.
        estimator_i:estimator in sklearn (default=0)
        the index of estimator.

        Returns
        -------
        y_predict

        """
        estimator = self.estimator[estimator_i]
        x0 = self.x0
        y0 = self.y0
        slices = list(slices)
        if len(slices) < 1:
            y_predict = np.zeros_like(y0)
        else:
            data_x0 = x0[:, slices]

            estimator.fit(data_x0, y0)
            if hasattr(estimator, 'best_estimator_'):
                estimator = estimator.best_estimator_
            else:
                pass
            y_predict = cross_val_predict(estimator, data_x0, y0, cv=5)

        return y_predict

    def score(self, slices, estimator_i=0):
        """

        Parameters
        ----------
        slices:list
        the index of X.
        estimator_i:estimator in sklearn (default=0)
        the index of estimator.

        Returns
        -------
         r2 score
        """
        y_pre = self.predict(list(slices), estimator_i)
        return metrics.r2_score(y_pre, self.y0)

    def cal_binary_score(self, slice1, slice2, estimator_i=0):

        set1 = set(slice1)
        set2 = set(slice2)
        set0 = set1 & set2
        y = [self.predict(list(i), estimator_i) for i in [set0, set1, set2]]
        score = metrics.r2_score(y[1] - y[0], y[2] - y[0])
        score = score if score >= 0 else 0
        return score

    def cal_binary_score_all(self, slices, n_jobs=1, estimator_i=0):
        cal_binary_distance = partial(self.cal_binary_score, estimator_i=estimator_i)
        if effective_n_jobs(n_jobs) == 1:
            parallel, func = list, cal_binary_distance
        else:
            parallel = Parallel(n_jobs=n_jobs)
            func = delayed(cal_binary_distance)
        slices_cuple = itertools.product(slices, repeat=2)
        scores = parallel(func(*slicesi) for slicesi in slices_cuple)
        scores = np.reshape(scores, (len(slices), len(slices)), order='F')
        return scores

    def score_all(self, slices, n_jobs=1, estimator_i=0):
        cal_score = partial(self.score, estimator_i=estimator_i)

        if effective_n_jobs(n_jobs) == 1:
            parallel, func = list, cal_score
        else:
            parallel = Parallel(n_jobs=n_jobs)
            func = delayed(cal_score)

        scores = parallel(func(slicesi) for slicesi in slices)
        return scores

    def group(self, slices, eps=0.1, estimator_i=0, printting=False):
        binary_score = self.cal_binary_score_all(slices, n_jobs=3, estimator_i=estimator_i)
        scores = 1 - binary_score
        db = DBSCAN(algorithm='auto', eps=eps, metric='precomputed',
                    metric_params=None, min_samples=2, n_jobs=None, p=None)
        db.fit(scores)
        label = db.labels_
        set_label = list(set(label))
        group = [[i for i in range(len(label)) if label[i] == aim] for aim in set_label]

        if printting:
            import networkx as nx
            g = nx.Graph()

            def my_ravel(data_cof):
                for i in range(data_cof.shape[0]):
                    for k in range(i + 1, data_cof.shape[0]):
                        yield i, k, data_cof[i, k]

            nodesize = [400] * scores.shape[0]
            distance_weight = list(my_ravel(scores))
            g.add_weighted_edges_from(distance_weight)
            # edges=nx.get_edge_attributes(g, 'weight').items()
            edges, weights = zip(*nx.get_edge_attributes(g, 'weight').items())
            pos = nx.layout.kamada_kawai_layout(g)
            lab = {i: i for i in range(len(slices))}
            nx.draw(g, pos, edgelist=edges, edge_color=np.around(weights, decimals=3) ** 0.5, labels=lab,
                    edge_cmap=plt.cm.Blues_r, edge_labels=nx.get_edge_attributes(g, 'weight'), edge_vmax=0.7,
                    node_color=np.array(label) + 1, vmin=0,
                    node_cmap=plt.cm.Accent, node_size=nodesize, width=weights,
                    )
            plt.show()
        return group

    def compare(self, score1, score2, theshold=0.01, greater_is_better=True):
        sign = 1 if greater_is_better else -1
        sig2 = 1 if score2 > 0 else -1
        if sign * score1 >= sign * (1 - sign * sig2 * theshold) * score2:
            return True
        else:
            return False

    def _select(self, slices, group, score, theshold=0.01, fliters=False, greater_is_better=True):
        score_group = [[score[i] for i in slicei_gruop] for slicei_gruop in group]
        select = [np.argmax(i) for i in score_group]  # 选择的在族中的位置
        for n, best, score_groupi, groupi in zip(range(len(select)), select, score_group, group):
            for i, _, index in zip(range(len(groupi)), score_groupi, groupi):
                if len(slices[groupi[best]]) > len(slices[index]) and self.compare(score_groupi[i], score_groupi[best],
                                                                                   theshold=theshold,
                                                                                   greater_is_better=greater_is_better):
                    best = i
            select[n] = best

        slices_select = [i[_] for _, i in zip(select, group)]  # 选择的在初始的位置
        slices_select = list(set(slices_select)) if fliters else slices_select
        score_select = [score[_] for _ in slices_select]  # 选择的分数
        selected = [slices[_] for _ in slices_select]  # 选择
        return score_select, selected

    def select(self, slices, group, estimator_i=0, theshold=0.01, greater_is_better=True):
        score = self.score_all(slices, n_jobs=3, estimator_i=estimator_i)
        return self._select(slices, group, score, theshold=theshold, fliters=False, greater_is_better=greater_is_better)


class MMGS(GS):
    """
    multi-model group select.
    """
    def __init__(self, estimator):
        super().__init__(estimator)

    def mmgroup(self, slices, eps=0.1):
        slices = [tuple(_) for _ in slices]

        group_result = [self.group(_, eps=eps, estimator_i=i, printting=False) for i, _ in enumerate(slices)]
        slice_gruop = []  # 分组位置
        for slicei in range(len(slices)):
            slicei_gruop = set()
            for group in group_result:
                for groupi in group:
                    if slicei in groupi:
                        slicei_gruop.update(groupi)
            slicei_gruop = list(slicei_gruop)
            slice_gruop.append(slicei_gruop)
        # todo hebing
        return slice_gruop

    def mmselect(self, slices, group, theshold=0.01, greater_is_better=True):
        score = [self.score_all(slices, n_jobs=3, estimator_i=i) for i in range(len(self.estimator))]
        score = np.mean(np.array(score), axis=0)
        std = np.std(np.array(score), axis=0)
        score = score - 0.5 * std
        return self._select(slices, group, score, theshold=theshold, fliters=True, greater_is_better=greater_is_better)


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
    # data = data225_import.values
    #
    # X = data[:, 2:]
    # y = data[:, 0]
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
    # gs.fit(X, y)
    # y = gs.predict(index[0][0])
    # y_score = gs.score(index[0][0])
    # y_banary = gs.cal_binary_score(index[6][0], index[5][0], estimator_i=0)
    #
    # index = [tuple(i) for i in list(zip(*index))[0][:20]]
    # # y_banary_all = gs.cal_binary_score_all(index, estimator_i=0)
    # group = gs.group(index, eps=0.2, printting=True)
    # select = gs.select(index, group, estimator_i=0, theshold=0.001, greater_is_better=True)
    # # index = [tuple(i) for i in list(zip(*index))[0][:10]]
    # # index2 = [tuple(i) for i in list(zip(*index2))[0][:10]]
    # # gs = MMGS([me1, me6])
    # # gs.fit(X, y)
    # # index_all = list(set(index) | set(index2))
    # # slice_gruop = gs.group_rank(index_all)
