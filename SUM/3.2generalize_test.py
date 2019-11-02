# -*- coding: utf-8 -*-

# @Time    : 2019/10/30 21:46
# @Email   : 986798607@qq.ele_ratio
# @Software: PyCharm
# @License: BSD 3-Clause


import warnings

import numpy as np
import pandas as pd
from sklearn import utils, preprocessing
from sklearn.model_selection import GridSearchCV

from featurebox.selection.score import dict_method_reg
from featurebox.tools.exports import Store
from featurebox.tools.imports import Call

warnings.filterwarnings("ignore")

"""
this is a description
"""

if __name__ == '__main__':
    store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp\3.sum')
    data = Call(r'C:\Users\Administrator\Desktop\band_gap_exp')
    data_import = data.csv.all_import
    name_init, abbr_init = data.csv.name_and_abbr

    select = ['destiny', 'distance core electron(schubert)', 'energy cohesive brewer', ]

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

    method_name = ["GPR-set", "SVR-set", 'KR-set', "KNR-set"]
    pre_y_all = []
    for i in method_name:
        me1, cv1, scoring1, param_grid1 = method = dict_method_reg()[i]

        estimator = GridSearchCV(me1, cv=cv1, scoring=scoring1, param_grid=param_grid1, n_jobs=1)
        estimator.fit(X, y)
        pre = estimator.predict(X)
        pre_y_all.append(pre)
    pre = np.array(pre_y_all)
