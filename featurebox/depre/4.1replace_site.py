# -*- coding: utf-8 -*-

# @Time   : 2019/6/8 21:35
# @Author : Administrator
# @Project : feature_preparation
# @FileName: symbollearing.py
# @Software: PyCharm

"""

"""
import itertools

import numpy as np
import pandas as pd
import sympy
from sklearn.metrics import explained_variance_score

from featurebox.combination.symbolbase import calculateExpr, getName
from featurebox.tools.exports import Store
from featurebox.tools.imports import Call

if __name__ == "__main__":
    store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp_last\4.symbollearning')
    data = Call(r'C:\Users\Administrator\Desktop\band_gap_exp_last\1.generate_data',
                r'C:\Users\Administrator\Desktop\band_gap_exp_last\3.MMGS',
                r'C:\Users\Administrator\Desktop\band_gap_exp_last\2.correction_analysis')

    all_import_structure = data.csv.all_import_structure
    data_import = all_import_structure
    data216_import = data_import.iloc[np.where(data_import['group_number'] == 216)[0]]
    data225_import = data_import.iloc[np.where(data_import['group_number'] == 225)[0]]
    data216_225_import = pd.concat((data216_import, data225_import))

    list_name = data.csv.list_name
    list_name = list_name.values.tolist()
    list_name = [[i for i in _ if isinstance(i, str)] for _ in list_name]
    # grid = itertools.product(list_name[2],list_name[12],list_name[32])
    grid = itertools.product(list_name[2],list_name[24],list_name[12])
    for select in grid:

        select = ['destiny'] + [j + "_%i" % i for j in select[1:] for i in range(2)]

        X_frame = data225_import[select]
        y_frame = data225_import['exp_gap']

        X = X_frame.values
        y = y_frame.values


        name, rep_name = getName(X_frame)
        x0, x1, x2, x3, x4 = rep_name
        # expr01 = sympy.log((x1**0.33 + x2*0.33) * x0**0.33 * x3**0.33 / x4**0.33 )
        expr01 = sympy.log(x3**1.0*x4**1.0*(x0**(-1.0))**(-1.0)*sympy.log(x1**2 + x2**2))

        results = calculateExpr(expr01, pset=None, x=X, y=y, score_method=explained_variance_score, add_coeff=True,
                                del_no_important=False, filter_warning=True, terminals=rep_name,
                                inter_add=True, iner_add=False, random_add=False)
        print(select)
        print(results)
