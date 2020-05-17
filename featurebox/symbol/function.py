#!/usr/bin/python
# coding:utf-8

"""
@author: wangchangxin
@contact: 986798607@qq.com
@software: PyCharm
@file: function.py
@time: 2020/5/15 22:06

Notes: the translation process
    the three function should be the same key.
    1.
    func_map(): repr of SymbolTree to sympy.Function
    func_map_dispose(): repr of SymbolTree to sympy.Function
    2.
    np_map(): repr of sympy.Function to numpy function
    3.
    dim_map(): repr of sympy.Function to Dim function
"""
import functools

import numpy as np
import sympy
from sympy import Function


def func_map():
    """str to sympy.Expr function"""

    def Div(left, right):
        return left / right

    def Sub(left, right):
        return left - right

    def zeroo(_):
        return 0

    def oneo(_):
        return 1

    def remo(ax):
        return 1 - ax

    functions2 = {"Add": sympy.Add, 'Sub': Sub, 'Mul': sympy.Mul, 'Div': Div}
    functions1 = {"sin": sympy.sin, 'cos': sympy.cos, 'exp': sympy.exp, 'log': sympy.ln,
                  'Abs': sympy.Abs, "Neg": functools.partial(sympy.Mul, -1.0),
                  "Rec": functools.partial(sympy.Pow, e=-1.0),
                  'Zeroo': zeroo, "Oneo": oneo, "Remo": remo}

    return functions1, functions2


def func_map_dispose():
    """user's str to sympy.expr function"""
    flat = Function("flat")
    return {"flat": flat, "Self": lambda x_: x_}


def np_map():
    """user's sympy.expr to np.ndarray function"""
    # flat = functools.partial(np.sum, axis=0)
    def flat(x):
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                return np.sum(x, axis=0)
            else:
                return x
        else:
            return x

    return {"flat": flat, "Self": lambda x_: x_}
