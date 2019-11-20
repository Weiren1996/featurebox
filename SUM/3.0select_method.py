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
from featurebox.tools.exports import Store
from featurebox.tools.imports import Call
from featurebox.tools.quickmethod import dict_method_reg
from featurebox.tools.show import BasePlot
from featurebox.tools.tool import name_to_name

warnings.filterwarnings("ignore")

"""
this is a description
"""
if __name__ == "__main__":
    store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp\3.sum\method')
    data = Call(r'C:\Users\Administrator\Desktop\band_gap_exp')
    data_import = data.csv.all_import
    name_init, abbr_init = data.name_and_abbr

    select = ['volume', 'destiny', 'lattice constants a', 'lattice constants c', 'radii covalent',
              'radii ionic(shannon)',
              'distance core electron(schubert)', 'latent heat of fusion', 'energy cohesive brewer', 'total energy',
              'charge nuclear effective(slater)', 'valence electron number', 'electronegativity(martynov&batsanov)',
              'volume atomic(villars,daams)']

    select = ['volume', 'destiny'] + [j + "_%i" % i for j in select[2:] for i in range(2)]

    data216_import = data_import.iloc[np.where(data_import['group_number'] == 216)[0]]
    data225_import = data_import.iloc[np.where(data_import['group_number'] == 225)[0]]
    data216_225_import = pd.concat((data216_import, data225_import))

    X_frame = data225_import[select]
    y_frame = data225_import['exp_gap']

    X = X_frame.values
    y = y_frame.values

    scal = preprocessing.MinMaxScaler()
    X = scal.fit_transform(X)
    X, y = utils.shuffle(X, y, random_state=5)

    from sklearn.model_selection import train_test_split

    X_train, _X_reserved_test, y_train, _y_reserved_test = train_test_split(X, y, test_size=0.25, random_state=2)

    X, y = X_train, y_train

    ###############
    method_name = 'GPR-set'
    me1, cv1, scoring1, param_grid1 = method = dict_method_reg()[method_name]

    estimator = GridSearchCV(me1, cv=cv1, scoring=scoring1, param_grid=param_grid1, n_jobs=1)

    n_select = (2, 3,)
    clf = Exhaustion(estimator, n_select=n_select, muti_grade=2, muti_index=[2, X.shape[1]], must_index=None,
                     n_jobs=4).fit(X, y)

    name_ = name_to_name(X_frame.columns.values, search=[i[0] for i in clf.score_ex[:10]], search_which=0,
                         return_which=(1,), two_layer=True)
    sc = np.array(clf.scatter)

    for i in clf.score_ex[:10]:
        print(i)
    for i in name_:
        print(i)

    t = clf.predict(X)
    p = BasePlot()
    p.scatter(y, t, strx='$E_{gap}$ true', stry='$E_{gap}$ calculated')
    plt.show()
    p.scatter(sc[:, 0], sc[:, 1], strx='number', stry='score')
    plt.show()

    store.to_csv(sc, method_name + "".join([str(i) for i in n_select]))
    store.to_pkl_pd(clf.score_ex, method_name + "".join([str(i) for i in n_select]))
