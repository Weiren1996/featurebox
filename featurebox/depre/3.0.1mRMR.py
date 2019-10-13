
from sklearn.model_selection import GridSearchCV
from featurebox.depre.mrmr import MutualInformationFeatureSelector
from featurebox.selection.score import dict_method_reg
from featurebox.tools.exports import Store
from featurebox.tools.imports import Call
import warnings
import pandas as pd
from sklearn import preprocessing, utils
import numpy as np

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


    feat_selector = MutualInformationFeatureSelector('MRMR',k=3, categorical=False,)
    # for i in index_all:
    #     result = feat_selector.fit(X[:,i], y)
    #     print(result._support_mask)
    result = feat_selector.fit(X[:,(1,1,8,9)], y)
    print(result._support_mask)
    # fea = [feature_selection.mutual_info_regression(X[:,i],y) for i in index_all]
