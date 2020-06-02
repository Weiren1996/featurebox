#!/usr/bin/python
# coding:utf-8

# @author: wangchangxin
# @contact: 986798607@qq.com
# @software: PyCharm
# @file: scores.py
# @License: GNU Lesser General Public License v3.0
"""
Notes:
    score method.
"""
import copy
import warnings

import numpy as np
import sympy
from scipy import optimize
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import r2_score
from sklearn.utils import check_array

from featurebox.symbol.dim import dim_map, dless, dnan
from featurebox.symbol.function import np_map
from featurebox.symbol.gp import compile_context, score_dim


# from featurebox.tools.tool import time_this_function


def addCoefficient(expr01, inter_add=True, inner_add=False):
    """
    try add coefficient to sympy expression.
    
    Parameters
    ----------
    expr01: Expr
        sympy expressions
    inter_add: bool
        bool
    inner_add: bool
        bool
        
    Returns
    -------
    expr
    """

    def get_args(expr_):
        """"""
        list_arg = []
        for i in expr_.args:
            list_arg.append(i)
            if i.args:
                re = get_args(i)
                list_arg.extend(re)

        return list_arg

    arg_list = get_args(expr01)
    arg_list = [i for i in arg_list if i not in expr01.args]
    cho = []
    cof_list = []

    if isinstance(expr01, sympy.Add):

        for i, j in enumerate(expr01.args):
            Wi = sympy.Symbol("W%s" % i)
            expr01 = expr01.subs(j, Wi * j)
            cof_list.append(Wi)

    else:
        A = sympy.Symbol("A")
        expr01 = sympy.Mul(expr01, A)
        cof_list.append(A)

    if inter_add:
        B = sympy.Symbol("B")
        expr01 = expr01 + B
        cof_list.append(B)

    if inner_add:
        cho_add = [i.args for i in arg_list if isinstance(i, sympy.Add)]
        cho_add = [[_ for _ in cho_addi if not _.is_number] for cho_addi in cho_add]
        [cho.extend(i) for i in cho_add]

        a_cho = [sympy.Symbol("k%s" % i) for i in range(len(cho))]
        for ai, choi in zip(a_cho, cho):
            expr01 = expr01.subs(choi, ai * choi)
        cof_list.extend(a_cho)

    return expr01, cof_list


def calculate_y(expr01, x, y, terminals, add_coef=True,
                filter_warning=True, inter_add=True, inner_add=False, np_maps=None):
    """
    calculate predict y by sympy expression.
    Parameters
    ----------
    expr01: Expr
        sympy expressions
    x: list of np.ndarray
        list of xi
    y: np.ndarray
        y value
    terminals: list of sympy.Symbol
        features and constants
    add_coef: bool
        bool
    filter_warning: bool
        bool
    inter_add: bool
        bool
    inner_add: bool
        bool
    np_maps: Callable
        user np.ndarray function

    Returns
    -------
    pre_y: 
        np.array or None
    expr01: Expr
        New expr.
    """
    if filter_warning:
        warnings.filterwarnings("ignore")
    if not np_maps:
        np_maps = np_map()

    expr00 = copy.deepcopy(expr01)

    if add_coef:

        expr01, a_list = addCoefficient(expr01, inter_add=inter_add, inner_add=inner_add)

        try:

            func0 = sympy.utilities.lambdify(terminals + a_list, expr01,
                                             modules=[np_maps, "numpy"])

            def func(x_, p):
                """"""
                num_list = []

                num_list.extend(x_)

                num_list.extend(p)
                return func0(*num_list)

            def res(p, x_, y_):
                """"""
                ress = y_ - func(x_, p)
                return ress

            result = optimize.least_squares(res, x0=[1.0] * len(a_list), args=(x, y),
                                            jac='3-point', loss='linear')
            cof = result.x

        except (ValueError, KeyError, NameError, TypeError):
            expr01 = expr00

        else:
            cof_ = []
            for a_listi, cofi in zip(a_list, cof):
                if "A" or "W" in a_listi.name:
                    cof_.append(cofi)
                else:
                    cof_.append(np.round(cofi, decimals=3))
            cof = cof_
            for ai, choi in zip(a_list, cof):
                expr01 = expr01.subs(ai, choi)
    try:
        func0 = sympy.utilities.lambdify(terminals, expr01, modules=[np_maps, "numpy"])
        re = func0(*x)
        re = re.ravel()
        assert y.shape == re.shape
        pre_y = check_array(re, ensure_2d=False)

    except (DataConversionWarning, AssertionError, ValueError, AttributeError, KeyError):
        pre_y = None

    return pre_y, expr01


def uniform_score(score_pen=1):
    """return the worse score"""
    if score_pen >= 0:
        return -np.inf
    elif score_pen <= 0:
        return np.inf
    elif score_pen == 0:
        return 0
    else:
        return score_pen


def calculate_score(expr01, x, y, terminals, scoring=None, score_pen=(1,), add_coef=True,
                    filter_warning=True, inter_add=True, inner_add=False, np_maps=None):
    """

    Parameters
    ----------
    expr01: Expr
        sympy expression.
    x: list of np.ndarray
        list of xi
    y: np.ndarray
        y value
    terminals: list of sympy.Symbol
        features and constants
    scoring: list of Callbale, default is [sklearn.metrics.r2_score,]
        See Also sklearn.metrics
    score_pen: tuple of  1 or -1
        1 : best is positive, worse -np.inf
        -1 : best is negative, worse np.inf
        0 : best is positive , worse 0
    add_coef: bool
        bool
    filter_warning: bool
        bool
    inter_add: bool
        bool
    inner_add: bool
        bool
    np_maps: Callable
        user np.ndarray function

    Returns
    -------
    score:float
        score
    expr01: Expr
        New expr.
    pre_y: np.ndarray or float
        np.array or None
    """
    if filter_warning:
        warnings.filterwarnings("ignore")
    if not scoring:
        scoring = [r2_score, ]
    if isinstance(score_pen, int):
        score_pen = [score_pen, ]

    assert len(scoring) == len(score_pen)

    pre_y, expr01 = calculate_y(expr01, x, y, terminals, add_coef=add_coef,
                                filter_warning=filter_warning, inter_add=inter_add, inner_add=inner_add,
                                np_maps=np_maps)

    try:
        sc_all = []
        for si, sp in zip(scoring, score_pen):
            sc = si(y, pre_y)
            if np.isnan(sc):
                sc = uniform_score(score_pen=sp)
            sc_all.append(sc)

    except (ValueError, RuntimeWarning):

        sc_all = [uniform_score(score_pen=i) for i in score_pen]

    return sc_all, expr01, pre_y


def calcualte_dim(expr01, terminals, dim_list, dim_maps=None):
    """

    Parameters
    ----------
    expr01: Expr
        sympy expression.
    terminals: list of sympy.Symbol
        features and constants
    dim_list: list of Dim
        dims of features and constants
    dim_maps: Callable
        user dim_maps

    Returns
    -------
    Dim:
        dimension
    dim_score
        is target dim or not
    """
    terminals = [str(i) for i in terminals]
    if not dim_maps:
        dim_maps = dim_map()
    func0 = sympy.utilities.lambdify(terminals, expr01, modules=[dim_maps])
    try:
        dim_ = func0(*dim_list)
    except ValueError:
        dim_ = dnan
    if isinstance(dim_, float):
        dim_ = dless

    return dim_


def calcualte_dim_score(expr01, terminals, dim_list, dim_type, fuzzy, dim_maps=None):
    """

    Parameters
    ----------
    expr01: Expr
        sympy expression.
    terminals: list of sympy.Symbol
        features and constants
    dim_list: list of Dim
        dims of features and constants
    dim_maps: Callable
        user dim_maps
    dim_type:list of Dim
        target dim
    fuzzy:
        fuzzy dim or not

    Returns
    -------
    Dim:
        dimension
    dim_score
        is target dim or not
    """
    dim_ = calcualte_dim(expr01, terminals, dim_list, dim_maps=dim_maps)

    dim_score = score_dim(dim_, dim_type, fuzzy)
    return dim_, dim_score


def calculate_collect(ind, context, x, y, terminals_and_constants_repr, dim_ter_con_list, dim_type, fuzzy, scoring=None,
                      score_pen=(1,),
                      add_coef=True, filter_warning=True, inter_add=True, inner_add=False, np_maps=None,
                      dim_maps=None, cal_dim=True):
    expr01 = compile_context(ind, context)

    score, expr01, pre_y = calculate_score(expr01, x, y, terminals_and_constants_repr,
                                           add_coef=add_coef, inter_add=inter_add,
                                           inner_add=inner_add,
                                           scoring=scoring, score_pen=score_pen,
                                           filter_warning=filter_warning,
                                           np_maps=np_maps)

    if cal_dim:
        dim, dim_score = calcualte_dim_score(expr01, terminals_and_constants_repr, dim_ter_con_list, dim_type, fuzzy,
                                             dim_maps=dim_maps)
    else:
        dim, dim_score = dless, 1

    return score, dim, dim_score
