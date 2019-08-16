
# -*- coding: utf-8 -*-

# @Time   : 2019/7/24 13:44
# @Author : Administrator
# @Project : feature_toolbox
# @FileName: gp.py
# @Software: PyCharm

import numpy as np
import pandas as pd
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor
from scipy.special._ufuncs import erf
from scipy.stats import pearsonr
from sklearn import utils

from featurebox.tools.exports import Store
from featurebox.tools.imports import Call

"""
this is a description
"""

def erff(a):
    return erf(a)

def my_custom_loss_func(y_true, y_pred, w):

    diff = 1-pearsonr(y_true, y_pred)[0]
    # diff = np.mean(diff)
    if np.isnan(diff):
        diff = 1
    return diff

mape = make_fitness(my_custom_loss_func, greater_is_better=False)

erff = make_function(function=erff,
                        name='erf',
                        arity=1)

est_gp = SymbolicRegressor(population_size=5000,
                           generations=10, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,verbose=1,
                           max_samples=0.9,function_set=("sub",'mul',"sin",erff),
                           parsimony_coefficient=0.1, random_state=0)
if __name__ == "__main__":

    store = Store(r'C:\Users\Administrator\Desktop\ma')
    data = Call(r'C:\Users\Administrator\Desktop\ma', index_col=None)
    data_import = data.csv.ma
    data_import = data_import.iloc[np.where(330<= data_import['t'] )[0]]
    data_import = data_import.iloc[np.where(2000>= data_import['t'] )[0]]
    data_import = data_import.iloc[np.where(2e-11>= data_import['k'] )[0]]
    x=data_import[["k","t"]]
    x[["k"]]=np.log10(x[["k"]])
    x=x.values
    y=data_import["c"].values
    #
    # scal = preprocessing.MinMaxScaler()
    # x = scal.fit_transform(x)

    # x, y = utils.shuffle(x, y, random_state=5)
    est_gp.fit(x, y)
    print(est_gp._program)
