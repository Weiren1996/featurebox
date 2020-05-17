#!/usr/bin/python
# coding:utf-8

"""
@author: wangchangxin
@contact: 986798607@qq.com
@software: PyCharm
@file: base.py
@time: 2020/5/14 13:31
"""
import copy
import functools
import warnings

import numpy as np
import sympy
from scipy import optimize
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import r2_score
from sklearn.utils import check_X_y
from sklearn.utils import check_array
from sympy import Expr

from featurebox.symbol.dim import dless, dim_map
from featurebox.symbol.function import func_map_dispose, func_map, np_map
from featurebox.symbol.gp import generate, genGrow, compile_


# from featurebox.tools.tool import time_this_function
# from featurebox.symbol import function
# from featurebox.symbol import gp


class SymbolTerminal:
    """General feature type.
    The name for show (str) and calculation (repr) are set to different string for avoiding repeated calculations.
    """
    __slots__ = ('name', 'value', 'arity', 'dim', "is_constant", "prob", 'conv_fct', "init_name")

    def __init__(self, values, name, dim=None, prob=None, init_name=None):
        """

        Parameters
        ----------
        values: number or np.ndarray
            xi values, the shape can be (n,) or (n_x,n)
        name: sympy.Symbol
            represent name
        dim: featurebox.symbol.dim.Dim or None
        prob: float or None
        init_name: str or None
            just for show, rather than calculate.
            Examples:
            init_name="[x1,x2]" , if is compact features, need[]
            init_name="(x1*x4-x3)", if is expr, need ()
        """
        if prob is None:
            prob = 1
        if dim is None:
            dim = dless
        self.value = values
        self.name = name
        self.conv_fct = str
        self.arity = 0
        self.dim = dim
        self.is_constant = False
        self.prob = prob
        self.init_name = init_name

    def format(self):
        """representing name"""
        return self.conv_fct(self.name)

    def format_init(self):
        """represented name"""
        if self.init_name:
            return self.conv_fct(self.init_name)
        else:
            return self.conv_fct(self.name.name)

    def __str__(self):
        """represented name"""
        if self.init_name:
            return self.init_name
        else:
            return self.name.name

    def __repr__(self):
        """represent name"""
        return self.name.name

    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot) for slot in self.__slots__)
        else:
            return NotImplemented

    def __hash__(self):
        return hash(str(self))


class SymbolConstant(SymbolTerminal):
    """General feature type."""

    def __init__(self, values, name, dim=None, prob=None):
        super(SymbolConstant, self).__init__(values, name, dim=dim, prob=prob)
        self.is_constant = True


class SymbolPrimitive:
    """General operation type"""
    __slots__ = ('name', 'func', 'arity', 'seq', 'prob', "args")

    def __init__(self, func, name, arity, prob=None):
        """
        Parameters
        ----------
        func: Callable
            Function. Better for sympy.Function Type.

            For Maintainer:
            If self function and can not be simplified to sympy.Function or elementary function,
            the function for function.np_map() and dim.dim_map() should be defined.
        name: str
            function name
        arity: int
            function input numbers
        prob: float
            default 1
        """
        if prob is None:
            prob = 1
        self.func = func
        self.name = str(name)
        self.arity = arity
        self.args = list(range(arity))
        self.prob = prob
        args = ", ".join(map("{{{0}}}".format, list(range(self.arity))))
        self.seq = "{name}({args})".format(name=self.name, args=args)

    def format(self, *args):
        return self.seq.format(*args)

    format_init = format  # for function the format for machine and user is the same.

    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot) for slot in self.__slots__)
        else:
            return NotImplemented

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return self.name

    __repr__ = __str__  # for function the format for machine and user is the same.


class SymbolSet(object):
    """
    Definite the operations, features,and fixed constants.
    """

    def __init__(self, name="SymbolSet"):
        self.arguments = []  # for translate
        self.name = name
        self.y = 0  # data y
        self.data_x = []  # data x
        self.new_num = 0

        self.primitives = []  # operation
        self.dispose = []  # structure operation
        self.terminals = []
        self.constants = []

        self.terms_count = 0
        self.prims_count = 0
        self.constant_count = 0
        self.dispose_count = 0

        self.dim_map = dim_map()
        self.np_map = np_map()

        self.prob_pri = {}  # probability of operation default is 1
        self.prob_dispose = {}  # probability of  structure operation, default is 1/n

        self.dim_ter_con = {}  # Dim of and features and constants
        self.prob_ter_con = {}  # probability of and features and constants

        self.context = {"__builtins__": None}  # all elements map
        self.mapping = dict()  # the same as deap.gp, but not use there.

        self.terminals_init_map = {}  # terminals representing name to represented name.

    def __repr__(self):
        return self.name

    __str__ = __repr__

    def _add_primitive(self, func, name, arity, prob=None, np_func=None, dim_func=None):

        """
        Parameters
        ----------
        name: str
            function name
        func: Callable
            function. Better for sympy.Function Type.
            If self function and can not be simplified to sympy.Function or elementary function,
            the function for np_func and dim_func should be defined.
        arity: int
            function input numbers
        prob: float
            default 1
        np_func: Callable
            numpy function or function constructed by numpy function
        dim_func: Callable
            function to calculate Dim
        """

        if prob is None:
            prob = 1

        if name is None:
            name = func.__name__

        assert name not in self.context, "Primitives are required to have a unique x_name. " \
                                         "Consider using the argument 'x_name' to rename your " \
                                         "second '%s' primitive." % (name,)
        if np_func:
            self.np_map[name] = np_func
            if dim_func is None:
                dim_func = lambda x: x
            self.dim_map[name] = dim_func

        self.prob_pri[name] = prob
        self.context[name] = func

        prim = SymbolPrimitive(func, name, arity, prob=prob)
        self.primitives.append(prim)
        self.prims_count += 1

    def _add_dispose(self, func, name, arity=1, prob=None, np_func=None, dim_func=None):
        """
        Parameters
        ----------
        name: str
            function name
        func: Callable
            function. Better for sympy.Function Type.
            If self function and can not be simplified to sympy.Function or elementary function,
            the function for np_func and dim_func should be defined.
        arity: 1
            function input numbers, must be 1
        prob: float
            default 1/n, n is structure function number.
        np_func: Callable
            numpy function or function constructed by numpy function
        dim_func: Callable
            function to calculate Dim
        """

        if prob is None:
            prob = 1

        if name is None:
            name = func.__name__

        assert name not in self.context, "Primitives are required to have a unique x_name. " \
                                         "Consider using the argument 'x_name' to rename your " \
                                         "second '%s' primitive." % (name,)
        if np_func:
            self.np_map[name] = np_func
            if dim_func is None:
                dim_func = lambda x: x
            self.dim_map[name] = dim_func

        self.prob_dispose[name] = prob
        self.context[name] = func

        prim = SymbolPrimitive(func, name, arity, prob=prob)
        self.dispose.append(prim)
        self.dispose_count += 1

    def _add_terminal(self, value, name, dim=None, prob=None, init_name=None):
        """
        Parameters
        ----------
        name: str
            function name
        value: numpy.ndarray
            xi value
        prob: float
            default 1
        init_name: str
            true name can be found of input. just for show, rather than calculate.
            Examples:
            init_name="[x1,x2]" , if is compact features, need[]
            init_name="(x1*x4-x3)", if is expr, need ()
        dim: Dim
            xi Dim
        """

        if prob is None:
            prob = 1
        if dim is None:
            dim = dless

        if name is None:
            name = "x%s" % self.terms_count

        assert name not in self.context, "Terminals are required to have a unique x_name. " \
                                         "Consider using the argument 'x_name' to rename your " \
                                         "second %s terminal." % (name,)

        assert "c0" not in self.context, "constant defination must be below to feature defination, " \
                                         "please add the feature first, ranther than constant"

        self.context[name] = sympy.Symbol(name)
        self.dim_ter_con[name] = dim
        self.prob_ter_con[name] = prob

        prim = SymbolTerminal(value, sympy.Symbol(name), dim=dim, prob=prob, init_name=init_name)
        self.data_x.append(value)
        self.terminals.append(prim)
        self.terms_count += 1

        if init_name:
            self.terminals_init_map[name] = init_name

    def _add_constant(self, value, name=None, dim=None, prob=None):
        """
        Parameters
        ----------
        name: str
            function name
        value: numpy.ndarray or float
            ci value
        prob: float
            default 1
        dim: Dim
            ci Dim
        """

        if prob is None:
            prob = 1
        if dim is None:
            dim = dless

        if name is None:
            name = "c%s" % self.constant_count

        assert name not in self.context, "Terminals are required to have a unique x_name. " \
                                         "Consider using the argument 'x_name' to rename your " \
                                         "second %s terminal." % (name,)

        self.context[name] = sympy.Symbol(name)
        self.dim_ter_con[name] = dim
        self.prob_ter_con[name] = prob

        prim = SymbolConstant(value, sympy.Symbol(name), dim=dim, prob=prob)

        self.data_x.append(value)
        self.constants.append(prim)
        self.constant_count += 1

    @property
    def terminalRatio(self):
        """Return the ratio of the number of terminals on the number of all
        kind of primitives.
        """
        return self.terms_count / float(self.terms_count + self.prims_count)

    @staticmethod
    def get_values(v, mean=False):
        """get list of dict values"""
        v = list(v.values())
        if mean:
            v = np.array(v)
            return list(v / sum(v))
        else:
            return v

    @property
    def prob_pro_ter_con_list(self):
        return self.get_values(self.prob_ter_con, mean=True)

    @property
    def prob_pri_list(self):
        return self.get_values(self.prob_pri, mean=True)

    @property
    def prob_dispose_list(self):
        return self.get_values(self.prob_dispose, mean=True)

    def compress(self):
        """Delete unnecessary detials, used before build Tree"""
        [delattr(i, "value") for i in self.terminals]
        [delattr(i, "value") for i in self.constants]
        [delattr(i, "dim") for i in self.terminals]
        [delattr(i, "dim") for i in self.constants]
        [delattr(i, "func") for i in self.dispose]
        [delattr(i, "func") for i in self.primitives]
        return self

    def add_operations(self,
                       power_categories=None,
                       categories=("Add", "Mul", "Self", "exp"),
                       partial_categories=None, self_categories=None):
        """

        Parameters
        ----------
        power_categories: None or list of float
            Examples:[0.5,2,3]
        categories: tuple of str
            map table:
                    {"Add": sympy.Add, 'Sub': Sub, 'Mul': sympy.Mul, 'Div': Div}
                    {"sin": sympy.sin, 'cos': sympy.cos, 'exp': sympy.exp, 'log': sympy.ln,
                      'Abs': sympy.Abs, "Neg": functools.partial(sympy.Mul, -1.0),
                      "Rec": functools.partial(sympy.Pow, e=-1.0), "sum": sum,
                      'Zeroo': zeroo, "Oneo": oneo, "Remo": remo, "Self": se}
        partial_categories: list of list
            just for dimensionless question!
            Examples:
                partial_categories = [["Add","Mul"],["x4"]]
        self_categories: list of list
            Examples:
                def rem(a):
                    return 1-a
                def rem_dim(d):
                    return d
                self_categories =  [['rem',rem, rem_dim, 1, 0.99]]
                                =  [['rem',rem, rem_dim, arity, prob]]
                                if rem_dim == None, apply default func, with return dim self

        Returns
        -------
        SymbolSet
        """
        if "Self" not in self.context:
            self.add_accumulative_operation()

        functions1, functions2 = func_map()
        if power_categories:
            for j, i in enumerate(power_categories):
                self._add_primitive(functools.partial(sympy.Pow, e=i),
                                    arity=1, name='pow%s' % j, prob=None)

        for i in categories:
            if i in functions1:
                self._add_primitive(functions1[i], arity=1, name=i, prob=None)
            if i in functions2:
                self._add_primitive(functions2[i], arity=2, name=i, prob=None)

        if partial_categories:
            for partial_categoriesi in partial_categories:
                for i in partial_categoriesi[0]:
                    for j in partial_categoriesi[1]:
                        if i in ["Mul", "Add"]:
                            self._add_primitive(functools.partial(functions2[i], sympy.Symbol(j)),
                                                name="{}({})".format(i, j), arity=1, prob=None)
                        elif i in ["Div", "Sub"]:
                            self._add_primitive(functools.partial(functions2[i], right=sympy.Symbol(j)),
                                                name="{}({})".format(i, j), arity=1, prob=None)
                        else:
                            pass
        if self_categories:
            for i in self_categories:
                assert len(i) == 5, "check your input of self_categories,wihch size must be 5"
                assert i[-2] == 1, "check your input of self_categories,wihch arity must be 1"
                self._add_primitive(sympy.Function(i[0]), arity=i[3], name=i[0], prob=i[4], np_func=i[1],
                                    dim_func=i[2])
        return self

    def add_accumulative_operation(self, categories=None, self_categories=None):
        """

        Parameters
        ----------
        categories: tuple of str
            categories=("flat","Self")
        self_categories: list of list
            Examples:
                def rem(ast):
                    return ast[0]+ast[1]+ast[2]
                def rem_dim(d):
                    return d
                self_categories = [['rem',rem, rem_dim, 1, 0.99]]
                                = [['rem',rem, rem_dim, arity, 0.99]]
                                if rem_dim == None, apply default func, with return dim self
                Note:
                the arity for accumulative_operation must be 1.
                if calculate of func rem relies on the size of ast,
                1.the size of each feature group is the same, such as n_gs.
                2.the size of ast must be the same as the size of feature group n_gs.

        Returns
        -------
        self
        """
        if not categories:
            categories = ["Self", "flat"]
        if isinstance(categories, str):
            categories = [categories, ]

        for i in categories:
            if i is "Self":
                self._add_dispose(func_map_dispose()[i], arity=1, name=i, prob=0.9)
            elif i is "flat":
                self._add_dispose(func_map_dispose()[i], arity=1, name=i, prob=0.1)
            else:
                self._add_dispose(func_map_dispose()[i], arity=1, name=i, prob=0.5)

        if self_categories:
            for i in self_categories:
                assert len(i) == 5, "check your input of self_categories,wihch size must be 5"
                assert i[-2] == 1, "check your input of self_categories,wihch arity must be 1"
                self._add_dispose(sympy.Function(i[0]), arity=i[3], name=i[0], prob=i[4], np_func=i[1], dim_func=i[2])

        return self

    def add_tree_to_features(self, Tree):
        """

        Parameters
        ----------
        Tree: SymbolTree
        Returns
        -------
        self
        """
        dim = Tree.dim
        init_name = str(Tree)
        value = Tree.pre_y

        # self.new_num
        name = "new%s" % self.new_num
        self.new_num += 1
        return self._add_terminal(self, value, name, dim=dim, prob=1, init_name=init_name)

    def add_features(self, X, y, feature_name=None, dim=1, prob=None, group=None):

        """

        Parameters
        ----------
        X: np.ndarray
            2D data
        y: np.ndarray
        feature_name: None, list of str
            the same size wih x.shape[1]
        dim: 1,list of Dim
            the same size wih x.shape[1]
        prob: None,list of float
            the same size wih x.shape[1]
        group: list of list
            features group

        Returns
        -------
        SymbolSet
        """
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        X, y = check_X_y(X, y)

        # define terminal
        n = X.shape[1]
        self.y = y.ravel()

        if dim is 1:
            dim = [dless for _ in range(n)]

        if prob is None:
            prob = [1 for _ in range(n)]

        if feature_name:
            assert n == len(dim) == len(feature_name) == len(prob)
            self.feature_name = feature_name

        else:
            assert n == len(dim) == len(prob)

        if not group:
            group = [[]]

        groups = []
        for i in group:
            groups.extend(i)

        for i, gi in enumerate(group):
            if len(gi) > 0:
                self._add_terminal(np.array(X.T[gi]),
                                   name="gx%s" % i, dim=dim[gi[0]], prob=prob[gi[0]])

        for i, (v, dimi, probi) in enumerate(zip(X.T, dim, prob)):
            if i not in groups:
                self._add_terminal(v, name="x%s" % i, dim=dimi, prob=probi)

        return self

    def add_constants(self, c, dim=1, prob=None):
        """

        Parameters
        ----------
        c: float, list of float
        dim: 1,list of Dim
            the same size wih c
        prob: None,list of float
            the same size wih c

        Returns
        -------
        SymbolSet
        """
        if isinstance(c, float):
            c = [c, ]

        n = len(c)

        if dim is 1:
            dim = [dless for _ in range(n)]

        if prob is None:
            prob = [1 for _ in range(n)]

        assert len(c) == len(dim) == len(prob)

        for v, dimi, probi in zip(c, dim, prob):
            self._add_constant(v, name=None, dim=dimi, prob=probi)

        return self


class _ExprTree(list):
    """
    Tree of expression
    """
    hasher = str

    def __init__(self, content):
        list.__init__(self, content)

    def __deepcopy__(self, memo):
        new = self.__class__(self)
        new.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return new

    def __setitem__(self, key, val):
        # Check for most common errors
        # Does NOT check for STGP constraints
        if isinstance(key, slice):
            if key.start >= len(self):
                raise IndexError("Invalid slice object (try to assign a %s"
                                 " in a tree of size %d). Even if this is allowed by the"
                                 " list object slice setter, this should not be done in"
                                 " the PrimitiveTree context, as this may lead to an"
                                 " unpredictable behavior for searchSubtree or evaluate."
                                 % (key, len(self)))
            total = val[0].arity
            for node in val[1:]:
                total += node.arity - 1
            if total != 0:
                raise ValueError("Invalid slice assignation : insertion of"
                                 " an incomplete subtree is not allowed in PrimitiveTree."
                                 " A tree is defined as incomplete when some nodes cannot"
                                 " be mapped to any position in the tree, considering the"
                                 " primitives' arity. For instance, the tree [sub, 4, 5,"
                                 " 6] is incomplete if the arity of sub is 2, because it"
                                 " would produce an orphan node (the 6).")
        elif val.arity != self[key].arity:
            raise ValueError("Invalid node replacement with a node of a"
                             " different arity.")

        list.__setitem__(self, key, val)

    def __str__(self):
        """Return the expression in a human readable string.
        """
        string = ""
        stack = []
        for node in self:
            if node.name == "Self":
                pass
            else:
                stack.append((node, []))
                while len(stack[-1][1]) == stack[-1][0].arity:
                    prim, args = stack.pop()
                    string = prim.format_init(*args)
                    if len(stack) == 0:
                        break  # If stack is empty, all nodes should have been seen
                    stack[-1][1].append(string)

        return string

    def __repr__(self):
        """Return the expression in a machine readable string for calculating.
        """
        string = ""
        stack = []
        for node in self:
            if node.name == "Self":
                pass
            else:
                stack.append((node, []))
                while len(stack[-1][1]) == stack[-1][0].arity:
                    prim, args = stack.pop()
                    string = prim.format(*args)
                    if len(stack) == 0:
                        break  # If stack is empty, all nodes should have been seen
                    stack[-1][1].append(string)

        return string

    @property
    def height(self):
        """Return the height of the tree, or the depth of the
        deepest node.
        """
        stack = [0]
        max_depth = 0
        for elem in self:
            depth = stack.pop()
            max_depth = max(max_depth, depth)
            stack.extend([depth + 1] * elem.arity)
        return max_depth / 2

    @property
    def root(self):
        """Root of the tree, the element 0 of the list.
        """
        return self[0]

    def searchSubtree(self, begin):
        """Return a slice object that corresponds to the
        range of values that defines the subtree which has the
        element with index *begin* as its root.
        """
        end = begin + 1
        total = self[begin].arity
        while total > 0:
            total += self[end].arity - 1
            end += 1
        return slice(begin, end)

    def __hash__(self):
        return hash(self.hasher(self))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def top(self):
        """accumulative operation"""
        return self[1::2]

    def bot(self):
        """operation and terminals"""
        return self[::2]


class SymbolTree(_ExprTree):
    """ Individual Tree, each tree is one expression"""

    def __init__(self, *arg, **kwargs):
        super(SymbolTree, self).__init__(*arg, **kwargs)
        self.p_name = [str(self), None]
        self.dim = dless
        self.pre_y = None
        self.expr = None

    def __setitem__(self, key, val):
        """keep these attribute refreshed"""
        self.p_name = [str(self), None]
        self.dim = dless
        self.pre_y = None
        self.expr = None
        [_ExprTree.__delattr__(self, i) for i in ("coef_expr", "coef_pre_y", "coef_score", "pure_expr", "pure_pre_y")]

        _ExprTree.__setitem__(self, key, val)

    def __getattribute__(self, item):
        """keep these attribute can be get only one time"""
        re = _ExprTree.__getattribute__(self, item)
        if item in ("coef_expr", "coef_pre_y", "coef_score", "pure_expr", "pure_pre_y"):
            _ExprTree.__delattr__(self, item)
        return re

    # @property
    # def terminals(self):
    #     """Return terminals that occur in the expression tree."""
    #     return [primitive for primitive in self if primitive.arity == 0]
    #
    # @property
    # def ter_site(self):
    #     return [i for i, primitive in enumerate(self) if primitive.arity == 0]
    #
    # @property
    # def primitives(self):
    #     """Return primitives that occur in the expression tree."""
    #     return [primitive for primitive in self if primitive.arity >= 1]
    #
    # @property
    # def pri_site(self):
    #     return [i for i, primitive in enumerate(self) if primitive.arity >= 1]

    def compile_(self, pset):
        """be not recommended for use"""
        return compile_(repr(self), pset)

    @classmethod
    def generate(cls, pset, min_, max_, condition, *kwargs):
        """details in generate function"""
        return cls(generate(pset, min_, max_, condition, *kwargs))

    @classmethod
    def genGrow(cls, pset, min_, max_):
        """details in genGrow function"""
        return cls(genGrow(pset, min_, max_))


def addCoefficient(expr01, inter_add=True, inner_add=False):
    """
    Parameters
    ----------
    expr01: Expr
    inter_add: bool
    inner_add: bool
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

    Parameters
    ----------
    expr01: Expr
    x: list of np.ndarray
        list of xi
    y: y
    terminals: list of sympy.Symbol
        features and constants
    add_coef: bool
    filter_warning: bool
    inter_add: bool
    inner_add: bool
    np_maps: Callable
        user np.ndarray function

    Returns
    -------
    pre_y: np.array or None
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
                return y_ - func(x_, p)

            result = optimize.least_squares(res, x0=[1] * len(a_list), args=(x, y),
                                            loss='linear', ftol=1e-3)
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
        # assert_all_finite(re)
        pre_y = check_array(re, ensure_2d=False)

    except (AttributeError, DataConversionWarning, AssertionError, ValueError,):
        # except (NameError, KeyError):
        pre_y = None

    return pre_y, expr01


def calculate_score(expr01, x, y, terminals, scoring=None, add_coef=True,
                    filter_warning=True, inter_add=True, inner_add=False, np_maps=None):
    """

    Parameters
    ----------
    expr01: Expr
    x: list of np.ndarray
        list of xi
    y: y
    terminals: list of sympy.Symbol
        features and constants
    scoring: Callbale, default is sklearn.metrics.r2_score
        See Also sklearn.metrics
    add_coef: bool
    filter_warning: bool
    inter_add: bool
    inner_add: bool
    np_maps: Callable
        user np.ndarray function

    Returns
    -------
    score:float
    expr01: Expr
        New expr.
    pre_y: np.array or None
    """
    if filter_warning:
        warnings.filterwarnings("ignore")
    if not scoring:
        scoring = r2_score

    pre_y, expr01 = calculate_y(expr01, x, y, terminals, add_coef=add_coef,
                                filter_warning=filter_warning, inter_add=inter_add, inner_add=inner_add,
                                np_maps=np_maps)

    try:
        score = scoring(y, pre_y)
        if np.isnan(score):
            score = -0

    except (ValueError):
        # except (ValueError, NameError, TypeError):
        score = -0

    return score, expr01, pre_y


def calcualte_dim(expr01, terminals, dim_list, dim_maps=None):
    """

    Parameters
    ----------
    expr01: Expr
    terminals: list of sympy.Symbol
        features and constants
    dim_list: list of Dim
        dims of features and constants
    dim_maps: Callable
        user dim_maps

    Returns
    -------
    Dim
    """
    terminals = [str(i) for i in terminals]
    if not dim_maps:
        dim_maps = dim_map()
    func0 = sympy.utilities.lambdify(terminals, expr01, modules=[dim_maps])
    dim_ = func0(*dim_list)
    return dim_


class CalculatePrecision:
    def __init__(self, pset, scoring=None, filter_warning=True):
        """

        Parameters
        ----------
        pset:SymbolSet
        scoring: Callbale, default is sklearn.metrics.r2_score
            See Also sklearn.metrics
        filter_warning:bool
        """
        self.pset = pset
        self.terminals = pset.terminals + pset.constants
        # list of sympy.Symbol, features and constants
        self.dim_x = pset.get_values(pset.dim_ter_con)  # list of dims
        self.data_x = pset.data_x  # list of xi
        self.dim_map = pset.dim_map
        self.np_map = pset.np_map

        self.y = pset.y  # list of y
        self.filter_warning = filter_warning
        self.scoring = scoring

    def get_expr(self, ind):
        """
        Parameters
        ----------
        ind: SymbolTree
        Returns
        -------
        Expr
        """
        if isinstance(ind, Expr):
            expr = ind
        else:
            expr = compile_(repr(ind), self.pset)
        return expr

    def calculate(self, ind=None, add_coef=True, inter_add=True, inner_add=False):
        """

        Parameters
        ----------
        ind: SymbolTree
        add_coef: bool
        inter_add: bool
        inner_add: bool

        Returns
        -------
        SymbolTree
        """
        if isinstance(ind, SymbolTree):
            expr = self.get_expr(self, ind)

            score, expr01, pre_y = calculate_score(expr, self.data_x, self.y, self.terminals,
                                                   add_coef=add_coef, inter_add=inter_add,
                                                   inner_add=inner_add,
                                                   scoring=self.scoring,
                                                   filter_warning=self.filter_warning,
                                                   np_maps=self.np_map)
            pure_pre_y, _ = calculate_y(expr, self.data_x, self.y, self.terminals,
                                        add_coef=False, inter_add=inter_add, inner_add=inner_add,
                                        filter_warning=self.filter_warning, np_maps=self.np_map)
            dim = calcualte_dim(expr, self.terminals, self.dim_x, self.dim_map)

            # this group should be get onetime and get all.
            ind.coef_expr = expr01
            ind.coef_pre_y = pre_y
            ind.coef_score = score
            ind.pure_expr = expr
            ind.pure_pre_y = pure_pre_y

            # add this attr for circle
            # see SymbolSet.add_Tree_to_feature
            ind.dim = dim
            ind.expr = expr
            ind.pre_y = pure_pre_y

        elif isinstance(ind, Expr):
            score, expr01, pre_y = calculate_score(ind, self.data_x, self.y, self.terminals,
                                                   add_coef=add_coef, inter_add=inter_add, inner_add=inner_add,
                                                   scoring=self.scoring,
                                                   filter_warning=self.filter_warning)
            expr01.score = score
            expr01.pre_y = pre_y
            return expr01

        return ind

    # if __name__ == "__main__":
    #     pset = SymbolSet()
    #     from sklearn.datasets import load_boston
    #
    #     data = load_boston()
    #     x = data["data"]
    #     t = data["target"]
    #     # pset.add_features(x, t, group=[[1, 2], [4, 5]])
    #     pset.add_features(x, t, group=None)
    #     pset.add_constants([6, 3], dim=[dless, dless], prob=None)
    #     pset.add_operations(power_categories=None,
    #                         categories=("Add", "Mul", "Self", "exp"),
    #                         partial_categories=None, self_categories=None)
    #
    #     cp = CalculatePrecision(pset)
    #
    #
    #     @time_this_function
    #     def run():
    #         z = 0
    #         for i in range(100):
    #             sl = SymbolTree.genGrow(pset, 3, 4)
    #             expr = cp.get_expr(sl)
    #             # a = calcualte_dim(expr, cp.terminals, cp.dim_x, cp.dim_map)
    #             # a = calculate_y(expr, cp.data_x, cp.y, cp.terminals,
    #             #                             add_coef=True)
    #             self = cp
    #             score, expr01, pre_y = calculate_score(expr, self.data_x, self.y, self.terminals,
    #                                                    add_coef=True, inter_add=False,
    #                                                    inner_add=True,
    #                                                    scoring=self.scoring,
    #                                                    filter_warning=self.filter_warning,
    #                                                    np_maps=self.np_map)
    # if a[0] is None:
    #     z+=1
    #     print(z)
    # else:
    #     print(a[0][:5])
    # print(score)

    # s = run()
    # a = calculate_y("flat(Abs(flat(gx0 + x8)))", cp.data_x, cp.y, cp.terminals,
    #                 add_coef=False)

    # random.seed(0)
    # sl = SymbolTree.genGrow(pset, 3, 4)
    # expr = cp.get_expr(sl)
    #
    # a = calcualte_dim(expr, cp.terminals, cp.dim_x, cp.dim_map)

    # sl = SymbolTree.genGrow(pset, 2, 3)
    # expr = compile_("flat(Mul(Add(x12, flat(x7)), Add(x10, gx1)))", pset)

    # cp = CalculatePrecision(pset)
    # cp.get_expr(expr)
    # cp.get_expr(sl)
