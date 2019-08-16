# -*- coding: utf-8 -*-

# @Time   : 2019/4/5 21:30
# @Author : Administrator
# @Project : feature-optimization-mutibest
# @FileName: cluster_and_plot.py
# @Software: PyCharm

"""
cluster the feature and plot
"""

import warnings
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import pairwise_distances

warnings.filterwarnings("ignore")


def cluster(data_t, eps=1, dra=False):

    distance_cof = pairwise_distances(data_t, metric='correlation')

    def my_ravel(data_cof):
        for i in range(data_cof.shape[0]):
            for k in range(i + 1, data_cof.shape[0]):
                yield i, k, data_cof[i, k]

    db = DBSCAN(algorithm='auto', eps=eps, metric='precomputed',
                metric_params=None, min_samples=2, n_jobs=None, p=None)
    db.fit(distance_cof)
    # c = db.core_sample_indices_
    label = db.labels_
    set_label = list(set(label))

    group = {"%s" % aim: [i for i in range(len(label)) if label[i] == aim] for aim in set_label}

    km = KMeans(n_clusters=1, random_state=0, tol=1e-3, max_iter=1000)
    core = []
    for i, j in list(group.items())[:-1]:
        if len(j) < 3:
            true_point = j[0]
        else:
            data_j = data_t[j]
            km.fit(data_t[j])
            point = km.cluster_centers_
            if len(point) > 1:
                point = point[0]
            dis = np.argmin(pairwise_distances(data_j, point, metric='correlation'))
            true_point = j[dis]
        core.append(true_point)
    core.sort()
    nodesize = [100] * distance_cof.shape[0]

    for i in core:
        nodesize[i] = nodesize[i] * 10
    if dra:
        g = nx.Graph()
        distance_weight = list(my_ravel(distance_cof))
        g.add_weighted_edges_from(distance_weight)
        # edges=nx.get_edge_attributes(g, 'weight').items()
        edges, weights = zip(*nx.get_edge_attributes(g, 'weight').items())
        edges, weights = [], []
        for _ in range(len(distance_weight)):
            if distance_weight[_][2] < 0.5:
                weights.append(distance_weight[_][2])
                edges.append(distance_weight[_][:2])

        pos = nx.layout.kamada_kawai_layout(g)
        # pos = nx.layout.spectral_layout(g)
        # scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
        # pca = PCA()
        # poss = pca.fit_transform(scaler.fit_transform(data_t.T).T)
        # poss= scaler.fit_transform(poss)[:, 0:2]
        # pos = {i: j for i, j in enumerate(poss)}

        nx.draw(g, pos, labels={i: i for i in core}, edgelist=edges, edge_color=np.around(weights, decimals=1) ** 0.5,
                edge_cmap=plt.cm.Blues_r, edge_labels=nx.get_edge_attributes(g, 'weight'), edge_vmax=0.7, node_color=np.array(label) + 1, vmin=0,
                node_cmap=plt.cm.tab20c, node_size=nodesize, width=weights,
                )

        plt.show()
    return group, core


if __name__ == "__main__":
    data = load_breast_cancer().data
    datat = data.T
    cluster(datat, eps=0.3, dra=True)
