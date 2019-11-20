# -*- coding: utf-8 -*-

# @Time    : 2019/11/12 16:10
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

import numpy as np
from sklearn.metrics import r2_score
from sklearn.utils import shuffle

from featurebox.combination.common import custom_loss_func
from featurebox.combination.dictbase import FixedSetFill
from featurebox.combination.dim import Dim
from featurebox.combination.symbolunit import mainPart
from featurebox.tools.exports import Store
from featurebox.tools.imports import Call
from featurebox.tools.tool import name_to_name

if __name__ == '__main__':
    import pandas as pd

    store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp\3.sum')
    data = Call(r'C:\Users\Administrator\Desktop\band_gap_exp')
    data_import = data.csv.all_import
    name_init, abbr_init = data.name_and_abbr

    select = ['destiny', 'energy cohesive brewer', 'distance core electron(schubert)']

    X_frame_abbr = name_to_name(name_init, abbr_init, search=select, search_which=1, return_which=2,
                                two_layer=False)

    select = ['destiny'] + [j + "_%i" % i for j in select[1:] for i in range(2)]

    select_abbr = ['$\\rho_c$'] + [j + "_%i" % i for j in X_frame_abbr[1:] for i in range(2)]

    data216_import = data_import.iloc[np.where(data_import['group_number'] == 216)[0]]
    data225_import = data_import.iloc[np.where(data_import['group_number'] == 225)[0]]
    data216_225_import = pd.concat((data216_import, data225_import))

    X_frame = data225_import[select]
    y_frame = data225_import['exp_gap']

    X = X_frame.values
    y = y_frame.values

    # scal = preprocessing.MinMaxScaler()
    # X = scal.fit_transform(X)
    X, y = shuffle(X, y, random_state=5)

    dim1 = Dim([0, -3, 0, 0, 0, 0, 0])
    dim2 = Dim([1, 2, -2, 0, 0, 0, 0])
    dim3 = Dim([1, 2, -2, 0, 0, 0, 0])
    dim4 = Dim([0, 1, 0, 0, 0, 0, 0])
    dim5 = Dim([0, 1, 0, 0, 0, 0, 0])
    target_dim = [Dim([1, 2, -2, 0, 0, 0, 0]), Dim([0, 0, 0, 0, 0, 0, 0])]

    dim_list = [dim1, dim2, dim3, dim4, dim5]

    # pset = ExpressionSetFill(x_name=select, power_categories=[2, 3], categories=("Add", "Mul", "exp"),
    #                          partial_categories=None, self_categories=None, dim=dim_list)

    pset = FixedSetFill(x_name=select_abbr, power_categories=[1 / 3, 1 / 2, 2, 3],
                        categories=('Add', 'Sub', 'Mul', 'Div', "Rec", 'exp', "log", "Self", "Abs", "Neg", "Rem"),
                        partial_categories=None, self_categories=None, dim=dim_list, max_=5,
                        definate_operate=[
                            [-17, [0, 1, 2, 3, "Abs", "Rec", 'exp', "log"]],
                            [-16, ['Mul', 'Div']],
                            [-15, ['Mul', 'Div']],

                            [-14, [0, 1, 2, 3, "Self", "log"]],
                            [-13, [0, 1, 2, 3, "Self", "Abs", "log"]],
                            [-12, [0, 1, 2, 3, "Self", "Abs", "log"]],

                            [-11, ["Self"]],
                            [-10, ["Self"]],
                            [-9, ["Rem"]],

                            [-8, ["Self"]],
                            [-7, ['Div', "Sub"]],
                            [-6, ['Div', "Sub"]],

                            [-5, [0, 1, 2, 3, "Self"]],
                            [-4, [0, 1, 2, 3, "Self"]],
                            [-3, [0, 1, 2, 3, "Self"]],

                            [-2, [0, 1, 2, 3, "Self"]],
                            [-1, [0, 1, 2, 3, "Self"]],

                        ],
                        definate_variable=[[-5, [0]],
                                           [-4, [1]],
                                           [-3, [2]],
                                           [-2, [3]],
                                           [-1, [4]]],
                        operate_linkage=[[-1, -2], [-3, -4]],
                        variable_linkage=None)
    result = mainPart(X, y, pset, pop_n=500, random_seed=2, cxpb=0.8, mutpb=0.5, ngen=20, tournsize=3, max_value=10,
                      double=False, score=[r2_score, custom_loss_func], target_dim=target_dim)
