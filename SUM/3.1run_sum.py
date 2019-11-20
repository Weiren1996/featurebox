import re
import warnings

import numpy as np
import pandas as pd
from sklearn import preprocessing, utils
from sklearn.model_selection import GridSearchCV

from featurebox.selection.sum import SUM
from featurebox.tools.exports import Store
from featurebox.tools.imports import Call
from featurebox.tools.quickmethod import dict_method_reg
from featurebox.tools.tool import name_to_name

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp\3.sum')
    data = Call(r'C:\Users\Administrator\Desktop\band_gap_exp\3.sum\method',
                r'C:\Users\Administrator\Desktop\band_gap_exp')
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
    # #
    X = X_frame.values
    y = y_frame.values

    scal = preprocessing.MinMaxScaler()
    X = scal.fit_transform(X)
    X, y = utils.shuffle(X, y, random_state=5)

    from sklearn.model_selection import train_test_split

    X_train, _X_reserved_test, y_train, _y_reserved_test = train_test_split(X, y, test_size=0.25, random_state=2)

    X, y = X_train, y_train

    """base_method"""
    method_name = ['GPR-set', 'SVR-set', 'KRR-set', 'KNR-set']
    index_all = [data.pickle_pd.GPR_set23, data.pickle_pd.SVR_set23, data.pickle_pd.KRR_set23, data.pickle_pd.KNR_set23]


    estimator_all = []
    for i in method_name:
        me1, cv1, scoring1, param_grid1 = dict_method_reg()[i]
        estimator_all.append(GridSearchCV(me1, cv=cv1, scoring=scoring1, param_grid=param_grid1, n_jobs=1))

    """union"""
    [print(_[0]) for _ in index_all]
    index_slice = [tuple(index[0]) for _ in index_all for index in _[:10]]
    index_slice = list(set(index_slice))

    """get x_name and abbr"""
    index_all_name = name_to_name(X_frame.columns.values, search=[i for i in index_slice],
                                  search_which=0, return_which=(1,), two_layer=True)

    index_all_name = [list(set([re.sub(r"_\d", "", j) for j in i])) for i in index_all_name]
    [i.sort() for i in index_all_name]
    index_all_abbr = name_to_name(name_init, abbr_init, search=index_all_name, search_which=1, return_which=2,
                                  two_layer=True)

    store.to_pkl_pd(index_slice, "index_all")
    store.to_csv(index_all_name, "index_all_name")
    store.to_csv(index_all_abbr, "index_all_abbr")

    """run"""
    self = SUM(estimator_all, index_slice, estimator_n=[0, 1, 2, 3], n_jobs=4)
    self.fit(X, y)

    c = self.cv_score_all(estimator_i=0)
    c1 = self.cv_score_all(estimator_i=1)
    c2 = self.cv_score_all(estimator_i=2)
    c3 = self.cv_score_all(estimator_i=3)
    print("max", np.max(c), np.argmax(c))
    print("max", np.max(c1), np.argmax(c1))
    print("max", np.max(c2), np.argmax(c2))
    print("max", np.max(c3), np.argmax(c3))

    # matrix_all = self.cal_binary_add_y_distance_all()
    # matrix = matrix_all[1]
    # self.cluster_print(matrix, label=None, eps=0.001, print_noise=0.001, node_name=None, highlight=[matrix.shape[0]-1])

    # md = self.y_distance_method()
    md = self.mean_max_method()

    # test2
    # res = []
    # for i in range(100):
    #
    #     X = X_frame.values
    #     y = y_frame.values
    #
    #     scal = preprocessing.MinMaxScaler()
    #     X = scal.fit_transform(X)
    #     X, y = utils.shuffle(X, y, random_state=5)
    #
    #     from sklearn.model_selection import train_test_split
    #
    #
    #     X_train, _X_reserved_test, y_train, _y_reserved_test = train_test_split(X, y, test_size=0.30, random_state=i)
    #
    #     X, y = X_train, y_train
    #
    #     """base_method"""
    #     method_name = ['GPR-set', 'SVR-set', 'KRR-set', 'KNR-set']
    #     index_all = [data.pickle_pd.GPR_set23, data.pickle_pd.SVR_set23, data.pickle_pd.KRR_set23,
    #                  data.pickle_pd.KNR_set23]
    #
    #     estimator_all = []
    #     for i in method_name:
    #         me1, cv1, scoring1, param_grid1 = dict_method_reg()[i]
    #         estimator_all.append(GridSearchCV(me1, cv=cv1, scoring=scoring1, param_grid=param_grid1, n_jobs=1))
    #
    #     """union"""
    #     [print(_[0]) for _ in index_all]
    #     index_slice = [tuple(index[0]) for _ in index_all for index in _[:10]]
    #     index_slice = list(set(index_slice))
    #
    #     """get x_name and abbr"""
    #     index_all_name = name_to_name(X_frame.columns.values, search=[i for i in index_slice],
    #                                   search_which=0, return_which=(1,), two_layer=True)
    #
    #     index_all_name = [list(set([re.sub(r"_\d", "", j) for j in i])) for i in index_all_name]
    #     [i.sort() for i in index_all_name]
    #     index_all_abbr = name_to_name(name_init, abbr_init, search=index_all_name, search_which=1, return_which=2,
    #                                   two_layer=True)
    #
    #     # store.to_pkl_pd(index_slice, "index_all")
    #     # store.to_csv(index_all_name, "index_all_name")
    #     # store.to_csv(index_all_abbr, "index_all_abbr")
    #
    #     """run"""
    #     self = SUM(estimator_all, index_slice, estimator_n=[0, 1, 2, 3], n_jobs=4)
    #     self.fit(X, y)
    #
    #     c = self.cv_score_all(estimator_i=0)
    #     c1 = self.cv_score_all(estimator_i=1)
    #     c2 = self.cv_score_all(estimator_i=2)
    #     c3 = self.cv_score_all(estimator_i=3)
    #     print("max", np.max(c), np.argmax(c))
    #     print("max", np.max(c1), np.argmax(c1))
    #     print("max", np.max(c2), np.argmax(c2))
    #     print("max", np.max(c3), np.argmax(c3))
    #
    #     # md = self.mean_max_method()
    #     md = self.y_distance_method()
    #
    #     result = [np.argmax(c), np.argmax(c1), np.argmax(c2), np.argmax(c3), md[0][0]]
    #
    #     res.append(result)
    # store.to_csv(res,"4_model_select_distance_resample")
