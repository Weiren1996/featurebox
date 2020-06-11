#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Time    : 2019/11/12 15:13
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: GNU Lesser General Public License v3.0

"""
Notes: the translation process
    the three function should be the same key.
    1.
    func_map(): repr of SymbolTree to sympy.Function
    func_map_dispose(): repr of SymbolTree to sympy.Function.
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

    def rem(ax):
        return 1 - ax

    functions1 = {"sin": sympy.sin, 'cos': sympy.cos, 'exp': sympy.exp, 'ln': sympy.ln,
                  'Abs': sympy.Abs, "Neg": functools.partial(sympy.Mul, -1.0),
                  "Rec": functools.partial(sympy.Pow, e=-1.0),"Self":lambda x:x,
                  "Rem": rem}
    functions2 = {"Add": sympy.Add, 'Sub': Sub, 'Mul': sympy.Mul, 'Div': Div}

    return functions1, functions2


def func_map_dispose():
    """user's str to sympy.expr function"""
    Flat = Function("MAdd")
    Comp = Function("MMul")
    Diff = Function("MSub")
    Quot = Function("MDiv")
    Self = lambda x: x
    Conv = Function("Conv")
    Flat.is_jump = False
    Comp.is_jump = False
    Diff.is_jump = True
    Quot.is_jump = True
    Conv.is_jump = True
    Self.is_jump = True

    Flat.keep = False
    Comp.keep = False
    Diff.keep = False
    Quot.keep = False
    Conv.keep = True
    Self.keep = True

    return {"MAdd": Flat, "MMul": Comp, "MSub": Diff, "MDiv": Quot, "Conv": Conv, "Self": Self}


def np_map():
    """user's sympy.expr to np.ndarray function"""

    def Flat(x):
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                return np.sum(x, axis=0)
            else:
                return x
        else:
            return x

    def Comp(x):
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                return np.prod(x, axis=0)
            else:
                return x
        else:
            return x

    def Diff(x):
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                if x.shape[0] == 2:
                    return x[0] - x[1]
                else:
                    return x
            else:
                return x
        else:
            return x

    def Quot(x):
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                if x.shape[0] == 2:
                    return x[0] / x[1]
                else:
                    return x
            else:
                return x
        else:
            return x

    def Conv(x):
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                if x.shape[0] == 2:
                    return np.array((x[1], x[0]))
                else:
                    return x
            else:
                return x
        else:
            return x

    return {"MAdd": Flat, "MMul": Comp, "MSub": Diff, "MDiv": Quot, "Conv": Conv,
            "Self": lambda x_: x_}