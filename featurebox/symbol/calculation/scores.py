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
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.utils import check_array

from featurebox.symbol.calculation.coefficient import try_add_coef
from featurebox.symbol.calculation.translate import compile_context
from featurebox.symbol.functions.dimfunc import dim_map, dless, dnan, Dim


def calculate_y(expr01, x, y, terminals, add_coef=True, x_test=None, y_test=None,
                filter_warning=True, inter_add=True, inner_add=False, vector_add=False, np_maps=None):
    if filter_warning:
        warnings.filterwarnings("ignore")
    try:
        if add_coef:
            pre_y, expr01 = try_add_coef(expr01, x, y, terminals,
                                         filter_warning=filter_warning,
                                         inter_add=inter_add, inner_add=inner_add,
                                         vector_add=vector_add, np_maps=np_maps)
        else:
            func0 = sympy.utilities.lambdify(terminals, expr01, modules=[np_maps, "numpy"])
            pre_y = func0(*x)

        if x_test is not None and y_test is not None:
            func0 = sympy.utilities.lambdify(terminals, expr01, modules=[np_maps, "numpy"])
            pre_y = func0(*x_test)
            pre_y = pre_y.ravel()
            assert y_test.shape == pre_y.shape
            pre_y = check_array(pre_y, ensure_2d=False)
        else:
            pre_y = pre_y.ravel()
            assert y.shape == pre_y.shape
            pre_y = check_array(pre_y, ensure_2d=False)

    except (DataConversionWarning, AssertionError, ValueError, AttributeError, KeyError, ZeroDivisionError):
        pre_y = None

    return pre_y, expr01


def calculate_y_unpack(expr01, x, terminals):
    try:
        func0 = sympy.utilities.lambdify(terminals, expr01)
        pre_y = func0(*x)
        pre_y = pre_y.ravel()
        pre_y = check_array(pre_y, ensure_2d=False)

    except (DataConversionWarning, AssertionError, ValueError, AttributeError, KeyError, ZeroDivisionError):
        pre_y = None
    return pre_y


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


def calculate_score(expr01, x, y, terminals, scoring=None, score_pen=(1,),
                    add_coef=True, filter_warning=True, inter_add=True,
                    inner_add=False, vector_add=False, np_maps=None):
    """

    Parameters
    ----------
    vector_add
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
    if not isinstance(scoring, (tuple, list)):
        scoring = [scoring, ]
    if isinstance(score_pen, int):
        score_pen = [score_pen, ]

    assert len(scoring) == len(score_pen), "the scoring and score_pen with same size"

    pre_y, expr01 = calculate_y(expr01, x, y, terminals, add_coef=add_coef,
                                filter_warning=filter_warning, inter_add=inter_add, inner_add=inner_add,
                                vector_add=vector_add,
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


def calculate_cv_score(expr01, x, y, terminals, scoring=None, score_pen=(1,), cv=5, refit=True,
                       add_coef=True, filter_warning=True, inter_add=True,
                       inner_add=False, vector_add=False, np_maps=None):
    """
    use cv spilt for score,return the mean_test_score.
    use cv spilt for predict,return the cv_predict_y.(not be used)
    Notes:
        if cv and refit, all the data is refit to determination the coefficients.
        Thus the expression is not compact with the this scores, when re-calculated by this expression
    Parameters
    ----------
    refit: True:
        use forced, refit the coefficient use all data.
    cv:sklearn.model_selection._split._BaseKFold,int
        the shuffler must be False
    vector_add
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

    if cv == 1:
        sc_all, expr01, pre_y = calculate_score(expr01, x, y, terminals, scoring=scoring, score_pen=score_pen,
                                                add_coef=add_coef, filter_warning=filter_warning, inter_add=inter_add,
                                                inner_add=inner_add, vector_add=vector_add, np_maps=np_maps)
        return sc_all, expr01, pre_y

    if isinstance(cv, int):
        cv = KFold(cv, shuffle=False)

    cv_sc_all = []
    cv_expr01 = []
    cv_pre_y = []
    xx = [xi for xi in x if isinstance(xi, np.ndarray)]
    c = [xi for xi in x if not isinstance(xi, np.ndarray)]
    xx = [xi.reshape((-1, 1)) if xi.ndim == 1 else xi.T for xi in xx]

    for train_index, test_index in cv.split(xx[0], y):

        X_train = [xi[train_index] for xi in xx]
        X_test = [xi[test_index] for xi in xx]
        y_train, y_test = y[train_index], y[test_index]

        X_train.reverse()
        X_test.reverse()
        nc = copy.deepcopy(c)
        nc.reverse()
        X_train = [X_train.pop() if isinstance(xi, np.ndarray) else nc.pop() for index, xi in enumerate(x)]
        nc = copy.deepcopy(c)
        nc.reverse()
        X_test = [X_test.pop() if isinstance(xi, np.ndarray) else nc.pop() for index, xi in enumerate(x)]

        pre_y, expr01 = calculate_y(expr01, X_train, y_train, terminals,
                                    x_test=X_test, y_test=y_test,
                                    add_coef=add_coef,
                                    filter_warning=filter_warning, inter_add=inter_add,
                                    inner_add=inner_add,
                                    vector_add=vector_add,
                                    np_maps=np_maps)

        try:
            sc_all = []
            for si, sp in zip(scoring, score_pen):
                sc = si(y_test, pre_y)
                if np.isnan(sc):
                    sc = uniform_score(score_pen=sp)
                sc_all.append(sc)

        except (ValueError, RuntimeWarning):

            sc_all = [uniform_score(score_pen=i) for i in score_pen]

        cv_sc_all.append(sc_all)
        cv_expr01.append(expr01)
        cv_pre_y.append(pre_y)

    sc_all = list(np.mean(np.array(cv_sc_all), axis=0))
    try:
        pre_y = np.concatenate(cv_pre_y)
    except ValueError:
        pre_y = None

    if refit is True:
        "the refit only use for see the detial of calcualtion after loop"
        sc_all0, expr01, pre_y0 = calculate_score(expr01, x, y, terminals, scoring=scoring, score_pen=score_pen,
                                                  add_coef=add_coef, filter_warning=filter_warning, inter_add=inter_add,
                                                  inner_add=inner_add, vector_add=vector_add, np_maps=np_maps)

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
        print(dim_type)
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
                      dim_ter_con_list, dim_type, fuzzy, cv=1, refit=True,
                      scoring=None, score_pen=(1,),
                      add_coef=True, filter_warning=True, inter_add=True, inner_add=False,
                      vector_add=False,
                      np_maps=None, dim_maps=None, cal_dim=True):
    expr01 = compile_context(ind, context, gro_ter_con)

    score, expr01, pre_y = calculate_cv_score(expr01, x, y, terminals_and_constants_repr,
                                              cv=cv, refit=refit,
                                              add_coef=add_coef, inter_add=inter_add,
                                              inner_add=inner_add, vector_add=vector_add,
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
