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
import sys
import warnings

import numpy as np
import sympy
from scipy import optimize
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import r2_score
from sklearn.utils import check_array
from sympy import Number, Function
from sympy.core.numbers import ComplexInfinity, NumberSymbol

from featurebox.symbol.dim import dim_map, dless, dnan, Dim
from featurebox.symbol.function import np_map, func_map_dispose, func_map


def compile_(expr, pset):
    """Compile the expression *expr*.

    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param pset: Primitive set against which the expression is compile.
    :returns: a function if the primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.
    """
    if isinstance(expr, str):
        code = expr
    else:
        code = repr(expr)
    if len(pset.arguments) > 0:
        # This section is a stripped version of the lambdify
        # function of SymPy 0.6.6.
        args = ",".join(arg for arg in pset.arguments)
        code = "lambda {args}: {code}".format(args=args, code=code)
    try:
        return eval(code, pset.context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError("DEAP : Error in tree evaluation :"
                          " Python cannot evaluate a tree higher than 90. "
                          "To avoid this problem, you should use bloat control on your "
                          "operators. See the DEAP documentation for more information. "
                          "DEAP will now abort.").with_traceback(traceback)


def simple(expr01, groups):
    """str to sympy.Expr function"""

    def calculate_number(expr):

        if isinstance(expr, sympy.Symbol):
            return expr, groups[expr.name]
        elif isinstance(expr, (Number, ComplexInfinity)):
            return expr, 1
        elif isinstance(expr, NumberSymbol):
            return expr, 1

        elif isinstance(expr, (Function, sympy.Expr)):

            if expr.func in func_map_dispose().values():
                ones = expr.func in [i for i in func_map_dispose().values() if not i.is_jump]
                if ones:
                    expr_arg, ns = calculate_number(expr.args[0])
                    if ns == 1:
                        expr = expr_arg
                    else:
                        expr = expr.func(expr_arg)
                        expr.conu = ns
                    return expr, 1

                else:
                    # """ jump = expr.func in func_map_dispose()["MSub", "MMDiv", "Conv"]"""
                    expr_arg, ns = calculate_number(expr.args[0])
                    if ns == 1:
                        expr = expr_arg
                    elif ns == 2:
                        expr = expr.func(expr_arg)
                        expr.conu = ns
                        ns = 1
                    else:
                        expr = expr_arg
                    return expr, ns

            elif expr.func in func_map()[1].values():
                new = [calculate_number(i) for i in expr.args]
                exprarg_new = list(zip(*new))[0]
                n = list(list(zip(*new))[1])
                expr = expr.func(*exprarg_new)
                n.append(1)
                le = len(set(n))
                if le >= 3:
                    return expr, np.nan
                else:
                    return expr, max(n)

            else:
                # """noch = expr.func in func_map()[0].values()"""
                return calculate_number(expr.args[0])
        else:
            raise TypeError(expr)

    expr01 = calculate_number(expr01)
    return expr01


def compile_context(expr, context, gro_ter_con):
    """Compile the expression *expr*.

    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param context: dict
    :param gro_ter_con: length
    :returns: a function if the primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.
    """
    if isinstance(expr, str):
        code = expr
    else:
        code = repr(expr)

    try:
        expr = eval(code, context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError("DEAP : Error in tree evaluation :"
                          " Python cannot evaluate a tree higher than 90. "
                          "To avoid this problem, you should use bloat control on your "
                          "operators. See the DEAP documentation for more information. "
                          "DEAP will now abort.").with_traceback(traceback)
    expr = simple(expr, gro_ter_con)[0]
    return expr


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
        for argi in expr_.args:
            list_arg.append(argi)
            if argi.args:
                re = get_args(argi)
                list_arg.extend(re)

        return list_arg

    cof_list = []

    if not inner_add:

        if isinstance(expr01, sympy.Add):
            for i, j in enumerate(expr01.args):
                Wi = sympy.Symbol("W%s" % i)
                expr01 = expr01.subs(j, Wi * j)
                cof_list.append(Wi)

        elif isinstance(expr01, (Function("MAdd"), Function("MMul"))):
            exprin1 = expr01.args[0]
            if isinstance(exprin1, sympy.Add):
                argss = []
                for i, j in enumerate(exprin1.args):
                    Wi = sympy.Symbol("W%s" % i)
                    argss.append(Wi * j)
                    cof_list.append(Wi)
                args_sum = sum(argss)
                expr01 = expr01.subs(exprin1, args_sum)

        else:
            A = sympy.Symbol("A")
            expr01 = sympy.Mul(expr01, A)
            cof_list.append(A)

    elif inner_add:
        arg_list = get_args(expr01)
        arg_list = [i for i in arg_list if i not in expr01.args]
        cho = []
        cho_add = [i.args for i in arg_list if isinstance(i, sympy.Add)]
        cho_add = [[_ for _ in cho_addi if not _.is_number] for cho_addi in cho_add]
        [cho.extend(i) for i in cho_add]

        a_cho = [sympy.Symbol("k%s" % i) for i in range(len(cho))]
        for ai, choi in zip(a_cho, cho):
            expr01 = expr01.subs(choi, ai * choi)
        cof_list.extend(a_cho)

    if inter_add:
        B = sympy.Symbol("B")
        expr01 = expr01 + B
        cof_list.append(B)

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
                                             modules=[np_maps, "numpy", "math"])

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

        except (ValueError, KeyError, NameError, TypeError, ZeroDivisionError):
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
