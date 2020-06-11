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

import warnings

import numpy as np
import sympy
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import r2_score
from sklearn.utils import check_array

from featurebox.symbol.calculation.coefficient import try_add_coef
from featurebox.symbol.calculation.dim import dim_map, dless, dnan, Dim
from featurebox.symbol.calculation.translate import compile_context


def calculate_y(expr01, x, y, terminals, add_coef=True,
                filter_warning=True, inter_add=True, inner_add=False, np_maps=None):
    if add_coef:
        expr01 = try_add_coef(expr01, x, y, terminals,
                              filter_warning=filter_warning, inter_add=inter_add, inner_add=inner_add, np_maps=np_maps)
    try:
        func0 = sympy.utilities.lambdify(terminals, expr01, modules=[np_maps, "numpy"])
        re = func0(*x)
        re = re.ravel()
        assert y.shape == re.shape
        pre_y = check_array(re, ensure_2d=False)

    except (DataConversionWarning, AssertionError, ValueError, AttributeError, KeyError, ZeroDivisionError):
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


def score_dim(dim_, dim_type, fuzzy=False):
    if dim_type is None:
        return 1
    elif isinstance(dim_type, str):
        if dim_type == 'integer':
            return 1 if dim_.isinteger() else 0
        elif dim_type == 'coef':
            return 1 if not dim_.anyisnan() else 0
        else:
            raise TypeError("dim_type should be None,'coef', 'integer', special Dim or list of Dim")
    elif isinstance(dim_type, list):
        return 1 if dim_ in dim_type else 0
    elif isinstance(dim_type, Dim):
        if fuzzy:
            return 1 if dim_.is_same_base(dim_type) else 0
        else:
            return 1 if dim_ == dim_type else 0

    else:
        raise TypeError("dim_type should be None,'coef','integer', special Dim or list of Dim")


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
    except (ValueError, TypeError, ZeroDivisionError):
        dim_ = dnan
    if isinstance(dim_, (float, int)):
        dim_ = dless
    if not isinstance(dim_, Dim):
        dim_ = dnan

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


def calculate_collect(ind, context, x, y, terminals_and_constants_repr, gro_ter_con,
                      dim_ter_con_list, dim_type, fuzzy,
                      scoring=None, score_pen=(1,),
                      add_coef=True, filter_warning=True, inter_add=True, inner_add=False,
                      np_maps=None, dim_maps=None, cal_dim=True):
    expr01 = compile_context(ind, context, gro_ter_con)

    score, expr01, pre_y = calculate_score(expr01, x, y, terminals_and_constants_repr,
                                           add_coef=add_coef, inter_add=inter_add,
                                           inner_add=inner_add,
                                           scoring=scoring, score_pen=score_pen,
                                           filter_warning=filter_warning,
                                           np_maps=np_maps)

    if cal_dim:
        dim, dim_score = calcualte_dim_score(expr01, terminals_and_constants_repr,
                                             dim_ter_con_list, dim_type, fuzzy,
                                             dim_maps=dim_maps)
    else:
        dim, dim_score = dless, 1

    return score, dim, dim_score