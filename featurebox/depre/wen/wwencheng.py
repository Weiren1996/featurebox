# -*- coding: utf-8 -*-

# @Time   : 2019/6/20 11:07
# @Author : Administrator
# @Project : feature_toolbox
# @FileName: wwencheng.py
# @Software: PyCharm
import functools
import os

import numpy as np
import pandas as pd
import sympy

from featurebox.combination.symbollearning import get_name, sympy_prim_set, main_part
from featurebox.tools.imports import Call

"""
this is a description
"""


if __name__ == '__main__':
    data = Call(r'C:\Users\Administrator\Desktop\wen20190615', backend="csv")
    os.chdir(r'C:\Users\Administrator\Desktop\wen20190615')
    BCC_C_G_X_2 = data.BCC_C_G_X_2
    BCC_G_Xmean_3 = data.BCC_G_Xmean_3
    BCC = BCC_C_G_X_2.join(BCC_G_Xmean_3)

    FCC_BCC_C_G_X_2 = data.FCC_BCC_C_G_X_2
    FCC_BCC_G_Xmean_3 = data.FCC_BCC_G_Xmean_3
    FCC_BCC = FCC_BCC_C_G_X_2.join(FCC_BCC_G_Xmean_3)

    FCC_C_G_X_2 = data.FCC_C_G_X_2
    FCC_G_Xmean_3 = data.FCC_G_Xmean_3
    FCC = FCC_C_G_X_2.join(FCC_G_Xmean_3)

    BCC_1 = data.BCC_G_deltG_deltX_1
    FCC_BCC_1 = data.FCC_BCC_G_deltG_deltX_1
    FCC_1 = data.FCC_G_deltG_deltX_1

    """#1一号问题"""

    def ques1():
        y = question_data["P"].values
        x_data = question_data.drop("P", axis=1)
        G = question_data["G"]
        x_data = x_data.drop("G", axis=1)
        x_data=x_data.join(G)
        x = x_data.values
        name, rep_name = get_name(x_data)
        pset1 = sympy_prim_set(
            categories=('Add', 'Sub', 'Mul', 'Div', 'exp', "Self"),
            name=name,
            partial_categories=[[["Mul"], ["x2"]]],
            rep_name=rep_name,
            index_categories=(1 / 5, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 1, 4 / 3, 3 / 2, 2, 3, 5),
            definate_operate=[

                (-4, ["Mul_x2"]),

                (-3, ['Add', 'Sub', 'Mul', 'Div']),
                (-2, [1, 2, 3, 4, 5, 6, 7, 8, 9, "exp"]),
                (-1, [1, 2, 3, 4, 5, 6, 7, 8, 9, "exp"]),
            ],
            definate_variable=None,
        )
        result = main_part(x, y, pset1, pop_n=500, random_seed=0, cxpb=0.8, mutpb=0.1, ngen=5, max_=2,
                           inter_add=None, iner_add=None, random_add=None)
        return result

    """#1一号问题结束"""
    """#2二号问题"""


    def ques2():
        y = question_data["P"].values
        question_data.drop("P", axis=1)
        index = []
        for i in zip(range(15), range(15, 30)):
            index.extend(i)
        for i in zip(range(30, 45), range(45, 60)):
            index.extend(i)
        x_data = question_data.iloc[:, index].join(question_data.iloc[:, -8:])
        x = x_data.values
        name, rep_name = get_name(x_data)

        linkage_operate = [
            list(range(1, 30, 2)),
            list(range(2, 31, 2)),
        ]
        linkage_operate2 = []
        for _ in range(4):
            linkage_operate2.extend([[j + _ * 30 for j in i] for i in linkage_operate])

        linkage_operate2.extend([
            list(range(121, 121 + 30)),
        ]
        )
        linkage = [[- _ - 60 for _ in i] for i in linkage_operate2]

        def funx(left, right):
            return 1 - left / right

        self_categories = [[sympy.Add, 15, "cont_Add"]]
        for i in ["x60", "x62", "x64", "x66", "x61", "x63", "x65", "x67"]:
            self_categories.append([functools.partial(funx, right=sympy.Symbol(i)), 1, "remDiv_{}".format(i)])
        pset1 = sympy_prim_set(
            categories=('Add', 'Sub', 'Mul', 'Div', 'exp', "Rem", "Rec", "Self"),
            name=name,
            rep_name=rep_name,
            partial_categories=[[["Mul"], ["x60", "x62", "x64", "x66"]]],
            index_categories=(1 / 5, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 1, 4 / 3, 3 / 2, 2, 3, 5),
            self_categories=self_categories,
            definate_operate=[

                (-156, ["Mul_x66", "Mul_x64", "Mul_x62", "Mul_x60"]),
                (-155, ['Add', 'Sub', 'Mul', 'Div']),

                (-154, [1, 2, 3, 4, 5, 6, 7, 8, 9, "exp"]),
                (-153, [1, 2, 3, 4, 5, 6, 7, 8, 9, "exp"]),

                (-152, ["cont_Add"]),
                (-151, ["cont_Add"]),

                (-121, ["Mul"]),

                (-92, ["Self"]),
                (-91, [1, 2, 3, 4, 5, 6, 7, 8, 9, "exp"]),
                (-62, ["Self"]),
                (-61, [1, 2, 3, 4, 5, 6, 7, 8, 9, "exp"]),

                (-32, [0, 1, 5, 9, 10]),
                (-31, ["remDiv_x66", "remDiv_x64", "remDiv_x62", "remDiv_x60"]),
                (-2, [0, 1, 5, 9, 10]),
                (-1, ["remDiv_x67", "remDiv_x65", "remDiv_x63", "remDiv_x61"]),
            ],
            definate_variable=None,
            linkage=linkage
        )
        result = main_part(x, y, pset1, pop_n=10, random_seed=7, cxpb=0.8, mutpb=0.1, ngen=5, max_=60,
                           inter_add=None, iner_add=None, random_add=None)
        return result


    """二号问题结束"""
    question_data = FCC_1
    resu = ques1()
    # question_data = FCC
    # resu = ques2()
