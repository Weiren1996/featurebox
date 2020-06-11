# -*- coding: utf-8 -*-

# @Time   : 2019/6/13 21:04
# @Author : Administrator
# @Project : feature_toolbox
# @FileName: 1.1add_compound_features.py
# @Software: PyCharm
import re
import warnings

import numpy as np
import pandas as pd
from sklearn import utils
from sklearn.model_selection import GridSearchCV

from featurebox.selection.quickmethod import dict_method_reg
from featurebox.selection.sum import SUM
from featurebox.tools.exports import Store
from featurebox.tools.imports import Call
from featurebox.tools.tool import name_to_name

warnings.filterwarnings("ignore")

"""
this is a description
"""
if __name__ == "__main__":

    import os

    os.chdir(r'band_gap')

    data = Call(os.getcwd(), "model")
    all_import = data.csv().all_import
    name_and_abbr = data.csv().name_and_abbr
    store = Store()

    data_import = all_import
    data225_import = data_import

    select = ['cell volume', 'cell density', 'lattice constants a', 'lattice constants c', 'covalent radii',
              'ionic radii(shannon)',
              'core electron distance(schubert)', 'fusion enthalpy', 'cohesive energy(Brewer)', 'total energy',
              'effective nuclear charge(slater)', 'valence electron number', 'electronegativity(martynov&batsanov)',
              'atomic volume(villars,daams)']  # human select

    select = ['cell volume', 'cell density', ] + [j + "_%i" % i for j in select[2:] for i in range(2)]

    X_frame = data225_import[select]
    y_frame = data225_import['exp_gap']

    X = X_frame.values
    y = y_frame.values

    X, y = utils.shuffle(X, y, random_state=5)

    """base_method"""

    method_name = ['GPR-set', 'SVR-set', 'KRR-set', 'KNR-set', 'GBR-em', 'AdaBR-em', 'RFR-em', "DTR-em"]

    index_all = [data.pickle_pd().GPR_set23, data.pickle_pd().SVR_set23, data.pickle_pd().KRR_set23]

    estimator_all = []
    for i in method_name:
        me1, cv1, scoring1, param_grid1 = dict_method_reg()[i]
        estimator_all.append(GridSearchCV(me1, cv=cv1, scoring=scoring1, param_grid=param_grid1, n_jobs=1))

    """union"""
    # [print(_[0]) for _ in index_all]
    index_slice = [tuple(index[0]) for _ in index_all for index in _[:round(len(_) / 3)]]
    # we choice top 30% for simpilfy calculation.
    index_slice = list(set(index_slice))
    index_slice.sort()

    # index_slice = [index_slice[i] for i in [
    # 127,
    # 151,
    # 135,
    # 126,
    # 121,
    # 116,
    # 168,
    # 119,
    # 122,
    # 123]]
    # # best 10 for set index for linux system.

    """get x_name and abbr"""
    index_all_name = name_to_name(X_frame.columns.values, search=[i for i in index_slice],
                                  search_which=0, return_which=(1,), two_layer=True)

    index_all_name = [list(set([re.sub(r"_\d", "", j) for j in i])) for i in index_all_name]
    [i.sort() for i in index_all_name]
    index_all_abbr = name_to_name(name_and_abbr.columns.values, list(name_and_abbr.iloc[0, :]), search=index_all_name,
                                  search_which=1, return_which=(2,),
                                  two_layer=True)

    parto = []
    table = []

    for i in range(2):
        print(i)
        X = X_frame.values
        y = y_frame.values

        # scal = preprocessing.MinMaxScaler()
        # X = scal.fit_transform(X)
        X, y = utils.shuffle(X, y, random_state=i)

        """run"""
        self = SUM(estimator_all, index_slice, estimator_n=[0, 1, 2, 3, 4, 5, 6, 7], n_jobs=12, batch_size=1)
        self.fit(X, y)
        mp = self.pareto_method()
        partotimei = list(list(zip(*mp))[0])

        tabletimei = np.vstack([self.resultcv_score_all_0, self.resultcv_score_all_1, self.resultcv_score_all_2,
                                self.resultcv_score_all_3, self.resultcv_score_all_4, self.resultcv_score_all_5,
                                self.resultcv_score_all_6, self.resultcv_score_all_7])

        parto.extend(partotimei)
        table.append(tabletimei)

    table = np.array(table)
    means_y = np.mean(table, axis=0).T
    result = pd.DataFrame(means_y)
    all_mean = np.mean(means_y, axis=1).T
    all_std = np.std(means_y, axis=1).T

    select_support = np.zeros(len(index_slice))
    mean_parto_index = self._pareto(means_y)
    select_support[mean_parto_index] = 1

    result["all_mean"] = all_mean
    result["all_std"] = all_std
    result["parto_support"] = select_support
    result['index_all_abbr'] = index_all_abbr
    result['index_all_name'] = index_all_name
    result['index_all'] = index_slice
    #
    result = result.sort_values(by="all_mean", ascending=False)
    store.to_csv(result, "feature_subset_rank")

    # for i in range():
