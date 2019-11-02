#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/30 18:10
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
this is a description
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, r2_score

from featurebox.combination.symbolbase import getName, sympyPrimitiveSet, mainPart
from featurebox.tools.exports import Store
from featurebox.tools.imports import Call

warnings.filterwarnings("ignore")
# if __name__ == "__main__":
#     store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp_last\4.symbollearning')
#     data_cluster = Call(r'C:\Users\Administrator\Desktop\band_gap_exp_last\1.generate_data',
#                 r'C:\Users\Administrator\Desktop\band_gap_exp_last\3.MMGS')
#
#     all_import = data_cluster.csv.all_import
#     data_import = all_import
#
#     select_gs = ['destiny', 'radii covalent', 'electronegativity(martynov&batsanov)']
#     select_gs = ['destiny'] + [j + "_%i" % i for j in select_gs[1:] for i in range(2)]
#
#     data216_import = data_import.iloc[np.where(data_import['group_number'] == 216)[0]]
#     data225_import = data_import.iloc[np.where(data_import['group_number'] == 225)[0]]
#     data216_225_import = pd.concat((data216_import, data225_import))
#
#     X_frame = data216_import[select_gs]
#     y_frame = data216_import['exp_gap']
#
#     X = X_frame.values
#     y = y_frame.values
#
#     # scal = preprocessing.MinMaxScaler()
#     # X = scal.fit_transform(X)
#     # X, y = utils.shuffle(X, y, random_state=5)
#
#     name, rep_name = getName(X_frame)
#     pset1 = sympyPrimitiveSet(rep_name=rep_name, types="fixed", max_=5,
#                               categories=('Add', 'Sub', 'Mul', 'Div', "Rec", 'exp', "log", "Abs", "Self", "Rem", "Neg"),
#                               power_categories=(1 / 3, 1 / 2, 1, 2, 3),
#                               definate_operate=[
#                                   [-11, [0, 1, 2, 3, 4, "Rec", 'exp', "log", "Abs"]],
#                                   [-10, ['Mul']],
#                                   [-9, ['Mul']],
#                                   [-8, ["Self"]],
#                                   [-7, ['Add', 'Sub', 'Mul', 'Div']],
#                                   [-6, ['Div']],
#
#                                   [-5, ["Rec", 'exp', "log","Self"]],
#                                   [-4, ["Rec", 'exp', "log","Self"]],
#                                   [-3, ["Rec", 'exp', "log","Self"]],
#
#                                   [-2, ["exp", "log","Rec", "Self"]],
#                                   [-1, ["exp", "log","Rec", "Self"]],
#                               ],
#                               definate_variable=[[-5, [0]],
#                                                  [-4, [1]],
#                                                  [-3, [2]],
#                                                  [-2, [3]],
#                                                  [-1, [4]]],
#                               operate_linkage=[[-1, -2], [-3, -4]],
#                               # variable_linkage = None
#                               )
#
#     result = mainPart(X, y, pset1, pop_n=500, random_seed=2, cxpb=0.8, mutpb=0.1, ngen=20,
#                        inter_add=True, iner_add=False, random_add=False, score=[explained_variance_score, r2_score])
#     ret = result[2][1]

# if __name__ == "__main__":
#     store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp_last\4.symbollearning')
#     data_cluster = Call(r'C:\Users\Administrator\Desktop\band_gap_exp_last\1.generate_data',
#                 r'C:\Users\Administrator\Desktop\band_gap_exp_last\3.MMGS')
#
#     all_import = data_cluster.csv.all_import
#     data_import = all_import
#
#     select_gs = ['destiny', 'enthalpy vaporization', 'electronegativity(martynov&batsanov)']
#     select_gs = ['destiny'] + [j + "_%i" % i for j in select_gs[1:] for i in range(2)]
#
#     data216_import = data_import.iloc[np.where(data_import['group_number'] == 216)[0]]
#     data225_import = data_import.iloc[np.where(data_import['group_number'] == 225)[0]]
#     data216_225_import = pd.concat((data216_import, data225_import))
#
#     X_frame = data225_import[select_gs]
#     y_frame = data225_import['exp_gap']
#
#     X = X_frame.values
#     y = y_frame.values
#
#     # scal = preprocessing.MinMaxScaler()
#     # X = scal.fit_transform(X)
#     # X, y = utils.shuffle(X, y, random_state=5)
#
#     name, rep_name = getName(X_frame)
#     pset1 = sympyPrimitiveSet(rep_name=rep_name, types="fixed", max_=5,
#                               categories=('Add', 'Sub', 'Mul', 'Div', "Rec", 'exp', "log", "Abs", "Self", "Rem", "Neg"),
#                               power_categories=(1 / 3, 1 / 2, 1, 2, 3),
#                               definate_operate=[
#                                   [-14, [0, 1, 2, 3, 4, "Rec", 'exp', "log", "Abs"]],
#                                   [-13, ['Mul']],
#                                   [-12, ['Mul']],
#
#                                   [-11, [0, 1, 2, 3, 4, "Rec", 'exp', "log"]],
#                                   [-10, [0, 1, 2, 3, 4, "Rec", 'exp', "log"]],
#                                   [-9, [0, 1, 2, 3, 4, "Rec", 'exp', "log"]],
#
#                                   [-8, ["Self"]],
#                                   [-7, ['Add', 'Sub', 'Mul', 'Div']],
#                                   [-6, ['Div']],
#
#                                   [-5, ["Rec", 'exp', "log","Self"]],
#                                   [-4, ["Rec", 'exp', "log","Self"]],
#                                   [-3, ["Rec", 'exp', "log","Self"]],
#
#                                   [-2, ["exp", "log","Rec", "Self"]],
#                                   [-1, ["exp", "log","Rec", "Self"]],
#                               ],
#                               definate_variable=[[-5, [0]],
#                                                  [-4, [1]],
#                                                  [-3, [2]],
#                                                  [-2, [3]],
#                                                  [-1, [4]]],
#                               operate_linkage=[[-1, -2], [-3, -4]],
#                               # variable_linkage = None
#                               )
#
#     result = mainPart(X, y, pset1, pop_n=500, random_seed=2, cxpb=0.8, mutpb=0.1, ngen=20,
#                        inter_add=True, iner_add=False, random_add=False, score=[explained_variance_score, r2_score])
#     ret = result[2][1]


# if __name__ == "__main__":
#     store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp_last\4.symbollearning')
#     data_cluster = Call(r'C:\Users\Administrator\Desktop\band_gap_exp_last\1.generate_data',
#                 r'C:\Users\Administrator\Desktop\band_gap_exp_last\3.MMGS')
#
#     all_import = data_cluster.csv.all_import
#     data_import = all_import
#
#     select_gs = ['destiny','electronegativity(martynov&batsanov)']
#     select_gs = ['destiny'] + [j + "_%i" % i for j in select_gs[1:] for i in range(2)]
#
#     data216_import = data_import.iloc[np.where(data_import['group_number'] == 216)[0]]
#     data225_import = data_import.iloc[np.where(data_import['group_number'] == 225)[0]]
#     data216_225_import = pd.concat((data216_import, data225_import))
#
#     X_frame = data225_import[select_gs]
#     y_frame = data225_import['exp_gap']
#
#     X = X_frame.values
#     y = y_frame.values
#
#     # scal = preprocessing.MinMaxScaler()
#     # X = scal.fit_transform(X)
#     # X, y = utils.shuffle(X, y, random_state=5)
#
#     name, rep_name = getName(X_frame)
#     pset1 = sympyPrimitiveSet(rep_name=rep_name, types="fixed", max_=3,
#                               categories=('Add', 'Sub', 'Mul', 'Div', "Rec", 'exp', "log", "Abs", "Self", "Rem", "Neg"),
#                               power_categories=(1 / 3, 1 / 2, 1, 2, 3),
#                               definate_operate=[
#                                   [-9, [0, 1, 2, 3, 4, "Rec", 'exp', "log", "Abs"]],
#                                   [-8, ['Mul']],
#
#                                   [-7, [0, 1, 2, 3, 4, "Rec", 'exp', "log"]],
#                                   [-6, [0, 1, 2, 3, 4, "Rec", 'exp', "log"]],
#
#                                   [-5, ["Self"]],
#
#                                   [-4, ['Div']],
#
#                                   [-3, ["Rec", 'exp', "log","Self"]],
#
#                                   [-2, ["exp", "log","Rec", "Self"]],
#                                   [-1, ["exp", "log","Rec", "Self"]],
#                               ],
#                               definate_variable=[
#                                                  [-3, [0]],
#                                                  [-2, [1]],
#                                                  [-1, [2]]],
#                               operate_linkage=[[-1, -2], ],
#                               # variable_linkage = None
#                               )
#
#     result = mainPart(X, y, pset1, pop_n=500, random_seed=2, cxpb=0.8, mutpb=0.1, ngen=20,
#                        inter_add=True, iner_add=False, random_add=False, score=[explained_variance_score, r2_score])
#     ret = result[2][1]

if __name__ == "__main__":
    store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp_last\4.symbollearning')
    data_cluster = Call(r'C:\Users\Administrator\Desktop\band_gap_exp_last\1.generate_data',
                        r'C:\Users\Administrator\Desktop\band_gap_exp_last\3.MMGS')

    all_import_structure = data_cluster.csv.all_import_structure
    data_import = all_import_structure

    select_gs = ['destiny', 'energy cohesive brewer', 'distance core electron(schubert)']
    select_gs = ['destiny'] + [j + "_%i" % i for j in select_gs[1:] for i in range(2)]

    data216_import = data_import.iloc[np.where(data_import['group_number'] == 216)[0]]
    data225_import = data_import.iloc[np.where(data_import['group_number'] == 225)[0]]
    data216_225_import = pd.concat((data216_import, data225_import))

    X_frame = data225_import[select_gs]
    y_frame = data225_import['exp_gap']

    X = X_frame.values
    y = y_frame.values

    # scal = preprocessing.MinMaxScaler()
    # X = scal.fit_transform(X)
    # X, y = utils.shuffle(X, y, random_state=5)

    name, rep_name = getName(X_frame)
    pset1 = sympyPrimitiveSet(rep_name=rep_name, types="fixed", max_=5,
                              categories=('Add', 'Sub', 'Mul', 'Div', "Rec", 'exp', "log", "Abs", "Self", "Rem", "Neg"),
                              power_categories=(1 / 3, 1 / 2, 1, 2, 3),
                              definate_operate=[
                                  [-14, [0, 1, 2, 3, 4, "Rec", 'exp', "log", "Abs"]],
                                  [-13, ['Mul']],
                                  [-12, ['Mul']],

                                  [-11, [0, 1, 2, 3, 4, "Rec", 'exp', "log"]],
                                  [-10, [0, 1, 2, 3, 4, "Rec", 'exp', "log"]],
                                  [-9, [0, 1, 2, 3, 4, "Rec", 'exp', "log"]],

                                  [-8, ["Self"]],
                                  [-7, ['Add', 'Sub', 'Mul', 'Div']],
                                  [-6, ['Add', 'Sub', 'Mul', 'Div']],

                                  [-5, ["Rec", 'exp', "log", "Self"]],
                                  [-4, [0, 1, 2, 3, 4, "Rec", 'exp', "log", "Self"]],
                                  [-3, [0, 1, 2, 3, 4, "Rec", 'exp', "log", "Self"]],

                                  [-2, [0, 1, 2, 3, 4, "exp", "log", "Rec", "Self"]],
                                  [-1, [0, 1, 2, 3, 4, "exp", "log", "Rec", "Self"]],
                              ],
                              definate_variable=[[-5, [0]],
                                                 [-4, [1]],
                                                 [-3, [2]],
                                                 [-2, [3]],
                                                 [-1, [4]]],
                              operate_linkage=[[-1, -2], [-3, -4]],
                              # variable_linkage = None
                              )

    result = mainPart(X, y, pset1, pop_n=500, random_seed=1, cxpb=0.8, mutpb=0.1, ngen=20,
                      inter_add=True, iner_add=False, random_add=False, score=[explained_variance_score, r2_score])
    ret = result[2][1]


# if __name__ == "__main__":
#     store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp_last\4.symbollearning')
#     data_cluster = Call(r'C:\Users\Administrator\Desktop\band_gap_exp_last\1.generate_data',
#                 r'C:\Users\Administrator\Desktop\band_gap_exp_last\3.MMGS')
#
#     all_import = data_cluster.csv.all_import
#     data_import = all_import
#
#     select_gs = ['destiny', 'energy cohesive brewer', 'volume atomic(villars,daams)']
#     select_gs = ['destiny'] + [j + "_%i" % i for j in select_gs[1:] for i in range(2)]
#
#     data216_import = data_import.iloc[np.where(data_import['group_number'] == 216)[0]]
#     data225_import = data_import.iloc[np.where(data_import['group_number'] == 225)[0]]
#     data216_225_import = pd.concat((data216_import, data225_import))
#
#     X_frame = data225_import[select_gs]
#     y_frame = data225_import['exp_gap']
#
#     X = X_frame.values
#     y = y_frame.values
#
#     # scal = preprocessing.MinMaxScaler()
#     # X = scal.fit_transform(X)
#     # X, y = utils.shuffle(X, y, random_state=5)
#
#     name, rep_name = getName(X_frame)
#     pset1 = sympyPrimitiveSet(rep_name=rep_name, types="fixed", max_=5,
#                               categories=('Add', 'Sub', 'Mul', 'Div', "Rec", 'exp', "log", "Abs", "Self", "Rem", "Neg"),
#                               power_categories=(1 / 3, 1 / 2, 1, 2, 3),
#                               definate_operate=[
#                                   [-14, [0, 1, 2, 3, 4, "Rec", 'exp', "log", "Abs"]],
#                                   [-13, ['Mul']],
#                                   [-12, ['Mul']],
#
#                                   [-11, [0, 1, 2, 3, 4, "Rec", 'exp', "log"]],
#                                   [-10, [0, 1, 2, 3, 4, "Rec", 'exp', "log"]],
#                                   [-9, [0, 1, 2, 3, 4, "Rec", 'exp', "log"]],
#
#                                   [-8, ["Self"]],
#                                   [-7, ['Add', 'Sub', 'Mul', 'Div']],
#                                   [-6, ['Add', 'Sub', 'Mul', 'Div']],
#
#                                   [-5, ["Rec", 'exp', "log","Self"]],
#                                   [-4, [0, 1, 2, 3, 4, "Rec", 'exp', "log","Self"]],
#                                   [-3, [0, 1, 2, 3, 4, "Rec", 'exp', "log","Self"]],
#
#                                   [-2, [0, 1, 2, 3, 4, "exp", "log","Rec", "Self"]],
#                                   [-1, [0, 1, 2, 3, 4, "exp", "log","Rec", "Self"]],
#                               ],
#                               definate_variable=[[-5, [0]],
#                                                  [-4, [1]],
#                                                  [-3, [2]],
#                                                  [-2, [3]],
#                                                  [-1, [4]]],
#                               operate_linkage=[[-1, -2], [-3, -4]],
#                               # variable_linkage = None
#                               )
#
#     result = mainPart(X, y, pset1, pop_n=500, random_seed=2, cxpb=0.8, mutpb=0.1, ngen=20,
#                        inter_add=True, iner_add=False, random_add=False, score=[explained_variance_score, r2_score])
#     ret = result[2][1]


# if __name__ == "__main__":
#     store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp_last\4.symbollearning')
#     data_cluster = Call(r'C:\Users\Administrator\Desktop\band_gap_exp_last\1.generate_data',
#                 r'C:\Users\Administrator\Desktop\band_gap_exp_last\3.MMGS')
#
#     all_import = data_cluster.csv.all_import
#     data_import = all_import
#
#     select_gs = ['destiny', 'energy cohesive brewer', 'radii covalent']
#     select_gs = ['destiny'] + [j + "_%i" % i for j in select_gs[1:] for i in range(2)]
#
#     data216_import = data_import.iloc[np.where(data_import['group_number'] == 216)[0]]
#     data225_import = data_import.iloc[np.where(data_import['group_number'] == 225)[0]]
#     data216_225_import = pd.concat((data216_import, data225_import))
#
#     X_frame = data225_import[select_gs]
#     y_frame = data225_import['exp_gap']
#
#     X = X_frame.values
#     y = y_frame.values
#
#     # scal = preprocessing.MinMaxScaler()
#     # X = scal.fit_transform(X)
#     # X, y = utils.shuffle(X, y, random_state=5)
#
#     name, rep_name = getName(X_frame)
#     pset1 = sympyPrimitiveSet(rep_name=rep_name, types="fixed", max_=5,
#                               categories=('Add', 'Sub', 'Mul', 'Div', "Rec", 'exp', "log", "Abs", "Self", "Rem", "Neg"),
#                               power_categories=(1 / 3, 1 / 2, 1, 2, 3),
#                               definate_operate=[
#                                   [-14, [0, 1, 2, 3, 4, "Rec", 'exp', "log", "Abs"]],
#                                   [-13, ['Mul']],
#                                   [-12, ['Mul']],
#
#                                   [-11, [0, 1, 2, 3, 4, "Rec", 'exp', "log"]],
#                                   [-10, [0, 1, 2, 3, 4, "Rec", 'exp', "log"]],
#                                   [-9, [0, 1, 2, 3, 4, "Rec", 'exp', "log"]],
#
#                                   [-8, ["Self"]],
#                                   [-7, ['Add', 'Sub', 'Mul', 'Div']],
#                                   [-6, ['Add', 'Sub', 'Mul', 'Div']],
#
#                                   [-5, ["Rec", 'exp', "log","Self"]],
#                                   [-4, [0, 1, 2, 3, 4, "Rec", 'exp', "log","Self"]],
#                                   [-3, [0, 1, 2, 3, 4, "Rec", 'exp', "log","Self"]],
#
#                                   [-2, [0, 1, 2, 3, 4, "exp", "log","Rec", "Self"]],
#                                   [-1, [0, 1, 2, 3, 4, "exp", "log","Rec", "Self"]],
#                               ],
#                               definate_variable=[[-5, [0]],
#                                                  [-4, [1]],
#                                                  [-3, [2]],
#                                                  [-2, [3]],
#                                                  [-1, [4]]],
#                               operate_linkage=[[-1, -2], [-3, -4]],
#                               # variable_linkage = None
#                               )
#
#     result = mainPart(X, y, pset1, pop_n=500, random_seed=2, cxpb=0.8, mutpb=0.1, ngen=20,
#                        inter_add=True, iner_add=False, random_add=False, score=[explained_variance_score, r2_score])
#     ret = result[2][1]

######
# if __name__ == "__main__":
#     store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp_last\4.symbollearning')
#     data = Call(r'C:\Users\Administrator\Desktop\band_gap_exp_last\1.generate_data',
#                 r'C:\Users\Administrator\Desktop\band_gap_exp_last\3.MMGS')
#
#     all_import = data.csv.all_import
#     data_import = all_import
#
#     select = ['destiny', 'energy cohesive brewer', 'electronegativity(martynov&batsanov)']
#     select = ['destiny'] + [j + "_%i" % i for j in select[1:] for i in range(2)]
#
#     data216_import = data_import.iloc[np.where(data_import['group_number'] == 216)[0]]
#     data225_import = data_import.iloc[np.where(data_import['group_number'] == 225)[0]]
#     data216_225_import = pd.concat((data216_import, data225_import))
#
#     X_frame = data225_import[select]
#     y_frame = data225_import['exp_gap']
#
#     X = X_frame.values
#     y = y_frame.values
#
#     # scal = preprocessing.MinMaxScaler()
#     # X = scal.fit_transform(X)
#     # X, y = utils.shuffle(X, y, random_state=5)
#
#     name, rep_name = getName(X_frame)
#     pset1 = sympyPrimitiveSet(rep_name=rep_name, types="fixed", max_=5,
#                               categories=('Add', 'Sub', 'Mul', 'Div', "Rec", 'exp', "log", "Abs", "Self", "Rem", "Neg"),
#                               power_categories=(1 / 3, 1 / 2, 1, 2, 3),
#                               definate_operate=[
#                                   [-14, [0, 1, 2, 3, 4, "Rec", 'exp', "log", "Abs"]],
#                                   [-13, ['Mul']],
#                                   [-12, ['Mul']],
#
#                                   [-11, [0, 1, 2, 3, 4, "Rec", 'exp', "log"]],
#                                   [-10, [0, 1, 2, 3, 4, "Rec", 'exp', "log"]],
#                                   [-9, [0, 1, 2, 3, 4, "Rec", 'exp', "log"]],
#
#                                   [-8, ["Self"]],
#                                   [-7, ['Add', 'Sub', 'Mul', 'Div']],
#                                   [-6, ['Add', 'Sub', 'Mul', 'Div']],
#
#                                   [-5, ["Rec", 'exp', "log","Self"]],
#                                   [-4, [0, 1, 2, 3, 4, "Rec", 'exp', "log","Self"]],
#                                   [-3, [0, 1, 2, 3, 4, "Rec", 'exp', "log","Self"]],
#
#                                   [-2, [0, 1, 2, 3, 4, "exp", "log","Rec", "Self"]],
#                                   [-1, [0, 1, 2, 3, 4, "exp", "log","Rec", "Self"]],
#                               ],
#                               definate_variable=[[-5, [0]],
#                                                  [-4, [1]],
#                                                  [-3, [2]],
#                                                  [-2, [3]],
#                                                  [-1, [4]]],
#                               operate_linkage=[[-1, -2], [-3, -4]],
#                               # variable_linkage = None
#                               )
#
#     result = mainPart(X, y, pset1, pop_n=100, random_seed=0, cxpb=0.8, mutpb=0.1, ngen=20,
#                       inter_add=True, iner_add=False, random_add=False, score=[explained_variance_score, r2_score])
#     ret = result[2][1]