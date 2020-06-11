"""
vector coef and vector const, which is a UndefinedFunction to excape the auto calculation of numpy to sympy.
"""
import copy
import warnings
from collections import Counter
from functools import reduce

import numpy as np
import sympy
from scipy import optimize
from sympy import Function
from sympy.core.function import UndefinedFunction


class Coef(UndefinedFunction):

    def __new__(mcs, name, arr):

        implementation = lambda x: arr * x
        f = super().__new__(mcs, name=name, _imp_=staticmethod(implementation))
        f.arr = arr
        f.name = name
        f.tp = "Coef"
        return f

    def __repr__(self):
        return str(self.arr)

    def __str__(self):
        return str(self.arr)

    def __eq__(self, other):
        if isinstance(other, Coef):
            return all(self.arr == other.arr)
        else:
            return False

    def __hash__(self):
        return hash((self.name, str(self)))


class Const(UndefinedFunction):

    def __new__(mcs, name, arr):

        implementation = lambda x: arr + x
        f = super().__new__(mcs, name=name, _imp_=staticmethod(implementation))
        f.arr = arr
        f.name = name
        f.tp = "Const"
        return f

    def __repr__(self):
        return str(self.arr)

    def __str__(self):
        return str(self.arr)

    def __eq__(self, other):
        if isinstance(other, Coef):
            return all(self.arr == other.arr)
        else:
            return False

    def __hash__(self):
        return hash((self.name, str(self)))


def get_args(expr, sole=True):
    def _get_args(expr_):
        """"""
        list_arg = []
        for argi in expr_.args:
            list_arg.append(argi)
            if argi.args:
                re = _get_args(argi)
                list_arg.extend(re)

        return list_arg

    list_a = _get_args(expr)
    if sole:
        count = Counter(list_a)
        term = []
        for i, j in count.items():
            if j == 1:
                term.append(i)
        list_a = term

    return list_a


def find_args(expr_, patten):
    """"""
    if len(expr_.args) > 0:
        if patten in expr_.args:
            return expr_.args
        else:
            for argi in expr_.args:
                d = find_args(argi, patten)
                if d is not None:
                    return d


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
    cof_list = []
    cof_dict = {}

    if isinstance(expr01, sympy.Add):
        for i, j in enumerate(expr01.args):
            Wi = sympy.Symbol("W%s" % i)
            expr01 = expr01.xreplace({j: Wi * j})
            cof_list.append(Wi)

    elif isinstance(expr01, (Function("MAdd"), Function("MSub"))):

        if hasattr(expr01, "conu") and expr01.conu > 1:
            Wi = sympy.Symbol("V")
            arg = expr01.args[0]
            expr02 = expr01.func(Wi * arg)
            if str(expr01) != str(expr02):
                cof_dict[Wi] = expr01.conu
            expr01 = expr02

        else:
            exprin1 = expr01.args[0]
            if isinstance(exprin1, sympy.Add):
                argss = []
                for i, j in enumerate(exprin1.args):
                    Wi = sympy.Symbol("W%s" % i)
                    argss.append(Wi * j)
                    cof_list.append(Wi)
                args_sum = sum(argss)
                expr01 = expr01.xreplace({exprin1: args_sum})

    else:
        A = sympy.Symbol("A")
        expr01 = sympy.Mul(expr01, A)
        cof_list.append(A)

    if inner_add:

        arg_list = get_args(expr01)

        arg_list = [i for i in arg_list if i not in expr01.args]
        cho = []
        cho_add = [i.args for i in arg_list if isinstance(i, sympy.Add)]
        cho_add = [[_ for _ in cho_addi if not _.is_number] for cho_addi in cho_add]
        [cho.extend(i) for i in cho_add]

        a_cho = [sympy.Symbol("k%s" % i) for i in range(len(cho))]

        for ai, choi in zip(a_cho, cho):
            expr02 = expr01.xreplace({choi: ai * choi})
            if str(expr01) != str(expr02):
                cof_list.append(ai)
            expr01 = expr02

        cho_add2 = [i for i in arg_list if isinstance(i, (Function("MAdd"), Function("MSub"))) if
                    hasattr(i, "conu") and i.conu > 1]

        for i, j in enumerate(cho_add2):

            Wi = sympy.Symbol("V%s" % i)
            arg = j.args[0]
            arg_new = j.func(Wi * arg)
            expr02 = expr01.xreplace({j: arg_new})
            if str(expr01) != str(expr02):
                cof_dict[Wi] = j.conu
            expr01 = expr02

    if inter_add:
        B = sympy.Symbol("B")
        expr01 = expr01 + B
        cof_list.append(B)

    return expr01, cof_list, cof_dict


class CheckCoef(object):
    def __init__(self, cof_list, cof_dict):
        """

        Parameters
        ----------
        cof_list:list
        cof_dict:dict
        """
        self.cof_list = cof_list
        self.cof_dict = cof_dict
        self.cof_dict_keys = list(cof_dict.keys())
        self.cof_dict_values = list(cof_dict.values())
        self.name = cof_list + list(self.cof_dict_keys)
        self.num = len(cof_list) + sum(list(self.cof_dict_values))

    def __len__(self):
        return len(self.name)

    @property
    def ind(self):
        lsa = list(range(len(self.cof_list)))
        n = len(lsa)
        for k in self.cof_dict_values:
            lsi = list(range(k))
            lsi = [lsii + n for lsii in lsi]
            lsa.append(lsi)
            n = lsi[-1] + 1

        return lsa

    def group(self, p, decimals=False):
        p = np.array(p)
        ls = []
        for i in self.ind:
            if isinstance(i, int):
                ls.append(p[i])
            else:
                ps = p[i].reshape((-1, 1))
                ls.append(ps)

        if decimals:
            cof_ = []
            for a_listi, cofi in zip(self.name, ls):

                if not isinstance(cofi, np.ndarray):
                    cof_.append(float("%.3e" % cofi))
                else:
                    cof_.append(np.array([float("%.3e" % i) for i in cofi]).reshape((-1, 1)))
            return cof_
        else:
            return ls


def try_add_coef(expr01, x, y, terminals,
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

    expr00 = copy.deepcopy(expr01)

    expr01, a_list, a_dict = addCoefficient(expr01, inter_add=inter_add, inner_add=inner_add)
    cc = CheckCoef(a_list, a_dict)
    try:

        func0 = sympy.utilities.lambdify(terminals + cc.name, expr01,  # short
                                         modules=[np_maps, "numpy", "math"])

        def func(x_, p):
            """"""
            num_list = []

            num_list.extend(x_)
            p = cc.group(p)

            num_list.extend(p)

            return func0(*num_list)

        def res(p, x_, y_):
            """"""
            ress = y_ - func(x_, p)
            return ress

        result = optimize.least_squares(res, x0=[1.0] * cc.num, args=(x, y),  # long
                                        jac='3-point', loss='linear')
        cof = result.x

    except (ValueError, KeyError, NameError, TypeError, ZeroDivisionError):
        # except (ValueError, KeyError,  ZeroDivisionError):
        expr01 = expr00

    else:
        cof = cc.group(cof, decimals=True)

        for ai, choi in zip(cc.name, cof):
            if ai in cc.cof_dict_keys:
                fun = Coef(ai.name, choi)
                olds0 = find_args(expr01, ai)
                if olds0 is None:
                    raise KeyError
                olds = [old for old in olds0 if old is not ai]
                if len(olds) == 1:
                    olds = olds[0]
                else:
                    olds = reduce(lambda x_, y_: x_ * y_, olds)
                expr01 = expr01.xreplace({ai * olds: fun(olds)})
            else:
                expr01 = expr01.xreplace({ai: choi})
    return expr01
