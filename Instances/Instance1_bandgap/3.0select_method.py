# -*- coding: utf-8 -*-

# @Time   : 2019/6/13 21:04
# @Author : Administrator
# @Project : feature_toolbox
# @FileName: 1.1add_compound_features.py
# @Software: PyCharm


import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import utils, preprocessing
from sklearn.model_selection import GridSearchCV

from featurebox.selection.exhaustion import Exhaustion
from featurebox.selection.quickmethod import dict_method_reg
from featurebox.tools.exports import Store
from featurebox.tools.imports import Call

from featurebox.tools.show import BasePlot
from featurebox.tools.tool import name_to_name

warnings.filterwarnings("ignore")

"""
this is a description
"""
if __name__ == "__main__":
    import os

    os.chdir(r'band_gap')

    data = Call()
    all_import = data.csv().all_import
    name_and_abbr = data.csv().name_and_abbr
    store = Store("model")

    data_import = all_import
    data225_import = data_import

    select = ['cell volume', 'electron density', 'lattice constants a', 'lattice constants c', 'covalent radii',
              'ionic radii(shannon)',
              'core electron distance(schubert)', 'fusion enthalpy', 'cohesive energy(Brewer)', 'total energy',
              'effective nuclear charge(slater)', 'valence electron number', 'electronegativity(martynov&batsanov)',
              'atomic volume(villars,daams)']  # human select

    select = ['cell volume', 'electron density', ] + [j + "_%i" % i for j in select[2:] for i in range(2)]

    X_frame = data225_import[select]
    y_frame = data225_import['exp_gap']

    X = X_frame.values
    y = y_frame.values

    X, y = utils.shuffle(X, y, random_state=5)

    ###############
    method_name = 'GPR-set'
    method_name = 'SVR-set'
    method_name = 'KNR-set'
    method_name = 'KRR-set'
    method_name = 'BRR-L1'
    method_name = 'PAR-L1'
    method_name = 'SGDR-L1'
    method_name = 'LASSO-L1'
    # method_name = 'AdaBR-em'
    # method_name = 'GBR-em'
    # method_name = 'DTR-em'
    # method_name = 'RFR-em'
    me1, cv1, scoring1, param_grid1 = method = dict_method_reg()[method_name]

    estimator = GridSearchCV(me1, cv=cv1, scoring=scoring1, param_grid=param_grid1, n_jobs=1)
    # n_select = [1,]
    n_select = (2, 3)
    # n_select = (2, 3, 4, 5)
    clf = Exhaustion(estimator, n_select=n_select, muti_grade=2, muti_index=[2, X.shape[1]], must_index=None,
                     n_jobs=1, refit=True).fit(X, y)

    name_ = name_to_name(X_frame.columns.values, search=[i[0] for i in clf.score_ex[:10]], search_which=0,
                         return_which=(1,), two_layer=True)
    sc = np.array(clf.scatter)

    for i in clf.score_ex[:]:
        print(i[1])
    for i in name_:
        print(i)

    t = clf.predict(X)
    p = BasePlot()
    p.scatter(y, t, strx='True $E_{gap}$', stry='Calculated $E_{gap}$')
    plt.show()
    p.scatter(sc[:, 0], sc[:, 1], strx='Number', stry='Score')
    plt.show()

    store.to_csv(sc, method_name + "".join([str(i) for i in n_select]))
    store.to_pkl_pd(clf.score_ex, method_name + "".join([str(i) for i in n_select]))