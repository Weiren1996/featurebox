# -*- coding: utf-8 -*-

# @Time   : 2019/6/13 21:04
# @Author : Administrator
# @Project : feature_toolbox
# @FileName: 1.1add_compound_features.py
# @Software: PyCharm


import pandas as pd
import numpy as np
from featurebox.selection.exhaustion import Exhaustion
from featurebox.selection.score import dict_method_reg
from featurebox.tools.exports import Store
from featurebox.tools.imports import Call
from featurebox.tools.show import BasePlot
from featurebox.tools.tool import index_to_name
import warnings
from sklearn import utils, kernel_ridge, gaussian_process, ensemble, linear_model, neighbors, preprocessing
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.linear_model import LogisticRegression, BayesianRidge, SGDRegressor, Lasso, ElasticNet, Perceptron
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_validate
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

"""
this is a description
"""

store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp_last\3.MMGS')
data = Call(r'C:\Users\Administrator\Desktop\band_gap_exp_last\1.generate_data')
all_import_structure = data.csv.all_import_structure
data_import = all_import_structure

select = ['volume', 'destiny', 'lattice constants a', 'lattice constants c', 'radii covalent', 'radii ionic(shannon)',
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

method_name = 'BayR-set'
me1, cv1, scoring1, param_grid1 = method = dict_method_reg()[method_name]

estimator = GridSearchCV(me1, cv=cv1, scoring=scoring1, param_grid=param_grid1, n_jobs=1)

clf = Exhaustion(estimator, n_select=(2, 3, 4, 5), muti_grade=2, muti_index=[2, X.shape[1]], must_index=None,
                 n_jobs=4).fit(X, y)

name = index_to_name([i[0] for i in clf.score_ex[:10]], X_frame.columns.values)

sc = np.array(clf.scatter)

import matplotlib.pyplot as plt

for i in clf.score_ex[:10]:
    print(i)

for i in name:
    print(i)

t = clf.predict(X)
p = BasePlot()
p.scatter(y, t, strx='$E_{gap}$ true', stry='$E_{gap}$ predict')
plt.show()
p.scatter(sc[:, 0], sc[:, 1], strx='number', stry='score')
plt.show()

store.to_csv(sc, method_name)
store.to_pkl_pd(clf.score_ex, method_name)
