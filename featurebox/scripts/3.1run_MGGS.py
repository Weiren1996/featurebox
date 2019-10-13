import re
import time

from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import GridSearchCV

from featurebox.selection.ugs import GS, UGS
from featurebox.selection.score import dict_method_reg
from featurebox.tools.exports import Store
from featurebox.tools.imports import Call
import warnings
import pandas as pd
from sklearn import preprocessing, utils
import numpy as np

from featurebox.tools.show import BasePlot
from featurebox.tools.tool import index_to_name

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp_last\3.MMGS\3.1')
    data = Call(r'C:\Users\Administrator\Desktop\band_gap_exp_last\1.generate_data',
                r'C:\Users\Administrator\Desktop\band_gap_exp_last\3.MMGS\3.0')

    all_import_structure = data.csv.all_import_structure
    data_import = all_import_structure

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

    method_name1 = 'GPR-set'
    method_name2 = 'SVR-set'
    method_name3 = 'KR-set'
    method_name4 = 'KNR-set'

    index1 = data.pickle_pd.GPR_set23
    index2 = data.pickle_pd.SVR_set23
    index3 = data.pickle_pd.KR_set23
    index4 = data.pickle_pd.KNR_set23

    index_all = [index1, index2, index3, index4]
    estimator_all = []
    for i in [method_name1, method_name2, method_name3, method_name4]:
        me1, cv1, scoring1, param_grid1 = dict_method_reg()[i]
        estimator_all.append(GridSearchCV(me1, cv=cv1, scoring=scoring1, param_grid=param_grid1, n_jobs=1))

    index_all = [tuple(index[0]) for _ in index_all for index in _[:10]]

    index_all = list(set(index_all))

    index_all_name = index_to_name([i for i in index_all], X_frame.columns.values)
    index_all_name2 = [list(set([re.sub("_\d", "", j) for j in i])) for i in index_all_name]
    [i.sort() for i in index_all_name2]


    def get_abbr(X_frame_name):
        element_table = pd.read_excel(r'F:\machine learning\feature_toolbox1.0\featurebox\data\element_table.xlsx',
                                      skiprows=0, index_col=0)
        name = list(element_table.loc["name"])
        abbr = list(element_table.loc["abbrTex"])
        name.extend(['face_dist1', 'vor_area1', 'face_dist2', 'vor_area2', "destiny", 'volume', "com"])
        abbr.extend(['$d_{vf1}$', '$S_{vf1}$', '$d_{vf2}$', '$S_{vf2}$', r"$\rho_c$", "$V_c$", "$com$"])
        index = [name.index(i) for i in X_frame_name]
        abbr = np.array(abbr)[index]
        return abbr

    index_all_abbr = [get_abbr(i) for i in index_all_name2]

    gs = UGS(estimator_all)
    gs.fit(X, y)
    re = gs.score_all(index_all, estimator_i=2)

    # slice_g = gs.cal_binary_distance_all(index_all, estimator_i=3, n_jobs=4)
    # groups = gs.cal_group(eps=0.10, estimator_i=3, printing=True, pre_binary_distance_all=slice_g, print_noise=1)
    # gs.cal_group(eps=0.10, estimator_i=1, printing=True, pre_binary_distance_all=slice_g, print_noise=0.1,node_name=index_all_abbr)
    # da = davies_bouldin_score(X.T, groups)
    # si = silhouette_score(X.T, groups)
    # import numpy as np
    # predict_y = [np.concatenate((y.reshape(-1,1), np.array([gs.predict(i, estimator_i=j) for i in index_all]).T),axis=1) for j in range(4)]
    # predict_y = [np.sort(i,axis=0) for i in predict_y]
    # [store.to_csv(predict_yi,"predict_y_sort_of_%s" %namei) for predict_yi,namei in zip(predict_y, [method_name1, method_name2, method_name3, method_name4])]


    # slice_group = gs.cal_t_group(index_all, eps=0.10)
    # present = gs.select_ugs(index_all, slice_group[1], theshold=0.1, greater_is_better=True)
    # present_name = index_to_name([i for i in present[1]], X_frame.columns.values)
    #
    # store.to_pkl_pd(slice_group, "cal_group of different model and tournment set")
    # store.to_pkl_pd(index_all, "index_all")
    #
    # store.to_csv(index_all_name, "index_all_name")
    # store.to_csv(index_all_abbr, "index_all_abbr")
    # store.to_csv(present_name, "present_name")
    #
    # present_score = np.array([gs.score_all(present[1], n_jobs=1, estimator_i=i) for i in range(len(estimator_all))]).T
    #
    # store.to_csv(present_score, "present_score")


