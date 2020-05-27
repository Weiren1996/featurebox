#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Time    : 2019/11/12 15:13
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: GNU Lesser General Public License v3.0

"""
Notes:
    this part is a customization from deap.
"""
import copy
import functools

import numpy as np
import sympy
from sklearn.utils import check_X_y, check_array

from featurebox.symbol.dim import dless, dim_map, dnan, Dim
from featurebox.symbol.function import func_map_dispose, func_map, np_map
from featurebox.symbol.gp import generate, genGrow, genFull, depart, compile_context
from featurebox.symbol.preference import PreMap
from featurebox.symbol.scores import calcualte_dim, calculate_score, calculate_collect
from featurebox.tools.tool import parallelize


class SymbolTerminal:
    """General feature type.
    The name for show (str) and calculation (repr) are set to different string for
    avoiding repeated calculations.
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
        self.name = str(name)
        self.conv_fct = str
        self.arity = 0
        self.dim = dim
        self.is_constant = False
        self.prob = prob
        self.init_name = init_name

    def format(self):
        # short,repr
        """representing name"""
        return self.conv_fct(self.name)

    def format_long(self):
        # long.str
        """represented name"""
        if self.init_name:
            return self.conv_fct(self.init_name)
        else:
            return self.conv_fct(self.name)

    def __str__(self):
        """represented name"""
        if self.init_name:
            return self.init_name
        else:
            return self.name

    def __repr__(self):
        """represent name"""
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(repr(self))


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

    format_long = format  # for function the format for machine and user is the same.

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        return self.name

    __repr__ = __str__  # for function the format for machine and user is the same.


class SymbolSet(object):
    """
    Definite the operations, features,and fixed constants.
    """

    def __init__(self, name="PSet"):
        self.arguments = []  # for translate
        self.name = name
        self.y = 0  # data y

        self.data_x_dict = {}  # data x
        self.y_dim = dless  # dim y

        self.new_num = 0

        self.terms_count = 0
        self.prims_count = 0
        self.constant_count = 0
        self.dispose_count = 0

        self.context = {"__builtins__": None}  # all elements map

        self.dim_map = dim_map()
        self.np_map = np_map()

        self.primitives_dict = {}
        self.prob_pri = {}  # probability of operation default is 1

        self.prob_dispose = {}  # probability of  structure operation, default is 1/n
        self.dispose_dict = {}

        self.dim_ter_con = {}  # Dim of and features and constants
        self.prob_ter_con = {}  # probability of and features and constants
        self.ter_con_dict = {}

        self.terminals_init_map = {}
        # terminals representing name "gx0" to represented name "[x1,x2]",
        # or "Mul(x2,x4)".
        self.terminals_fea_map = {}  # terminals Latex feature name.

        self.premap = PreMap.from_shape(3)

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
        self.primitives_dict[name] = prim
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
        self.dispose_dict[name] = prim
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

        self.context[name] = sympy.Symbol(name)
        self.dim_ter_con[name] = dim
        self.prob_ter_con[name] = prob

        prim = SymbolTerminal(value, sympy.Symbol(name), dim=dim, prob=prob, init_name=init_name)
        self.data_x_dict[name] = value
        self.ter_con_dict[name] = prim
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
            default 0.5
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

        self.data_x_dict[name] = value
        self.ter_con_dict[name] = prim
        self.constant_count += 1

    def add_operations(self, power_categories=None, categories=("Add", "Mul", "Self", "exp"),
                       self_categories=None, power_categories_prob="balance",
                       categories_prob="balance", special_prob=None):
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
                    "Rec": functools.partial(sympy.Pow, e=-1.0),

                    Others:
                    'Zeroo': zeroo
                     f(x)=0,if x true
                    "Oneo": oneo,
                     f(x)=1,if x true
                    "Remo": remo,
                     f(x)=1-x,if x true
                    "Self": se
                     f(x)=x,if x true
                     "Relu":
                     f(x)=x,if x>0
                     f(x)=0,if x<=0
                     }
        self_categories: list of list
            Examples:
                def rem(a):
                    return 1-a
                def rem_dim(d):
                    return d
                self_categories =  [['rem',rem, rem_dim, 1, 0.99]]
                                =  [['rem',rem, rem_dim, arity, prob]]
                                when rem_dim == None, (can beused when dont calculate dim),
                                would apply default func, with return dim self
        power_categories_prob:" balance" or float (0,1]
            probability of power categories,"balance" is 1/n_power_cat
        categories_prob: "balance" or float (0,1]
          probabilityty of categories, except (+,-*,/), "balance" is 1/n_categories.
            Notes: the  (+,-*,/) are set as 1 to be a standard.
        special_prob: None or dict
            Examples: {"Mul":0.6,"Add":0.4,"exp":0.1}
        Returns
        -------
        SymbolSet
        """

        def change(n, p):
            if isinstance(self_categories, dict):
                if n in special_prob:
                    p = special_prob[n]
            return p

        if "MAdd" not in self.context:
            self.add_accumulative_operation()

        functions1, functions2 = func_map()
        if power_categories:
            if power_categories_prob is "balance":
                prob = 1 / len(power_categories)
            elif isinstance(power_categories_prob, float):
                prob = power_categories_prob
            else:
                raise TypeError("power_categories_prob accept int from (0,1] or 'balance'.")
            for j, i in enumerate(power_categories):
                name = 'pow%s' % j
                prob = change(name, prob)
                self._add_primitive(functools.partial(sympy.Pow, e=i),
                                    arity=1, name='pow%s' % j, prob=prob)

        for i in categories:
            if categories_prob is "balance":
                prob1 = 1 / len([_ for _ in power_categories if _ not in ("Add", 'Sub' 'Mul', 'Div')])
            elif isinstance(categories_prob, float):
                prob1 = categories_prob
            else:
                raise TypeError("categories_prob accept int from (0,1] or 'balance'.")
            if i in functions1:
                prob1 = change(i, prob1)
                self._add_primitive(functions1[i], arity=1, name=i, prob=prob1)
            if i in functions2:
                prob2 = change(i, 1)
                self._add_primitive(functions2[i], arity=2, name=i, prob=prob2)

        if self_categories:
            for i in self_categories:
                assert len(i) == 5, "check your input of self_categories,wihch size must be 5"
                assert i[-2] == 1, "check your input of self_categories,wihch arity must be 1"
                prob = change(i[0], i[4])
                self._add_primitive(sympy.Function(i[0]), arity=i[3], name=i[0], prob=prob, np_func=i[1],
                                    dim_func=i[2])
        return self

    def add_accumulative_operation(self, categories=None, categories_prob="balance",
                                   self_categories=None, special_prob=None):
        """

        Parameters
        ----------
        categories: tuple of str
            categories=("Self","MAdd","MSub", "MMul","MDiv")
        categories_prob: "balance" or float (0,1]
            probility of categories, except ("Self","MAdd", "MSub", "MMul", "MDiv"),
            "balance" is 1/n_categories.
             "MSub", "MMul", "MDiv" only work on the size of group is 2, else work like "Self".
            Notes: the  ("Self","MAdd","MSub", "MMul", "MDiv") are set as 1 and 0.1 to be a standard.
        self_categories: list of list
            Examples:
                def rem(ast):
                    return ast[0]+ast[1]+ast[2]
                def rem_dim(d):
                    return d
                self_categories = [['rem',rem, rem_dim, 1, 0.99]]
                                = [['rem',rem, rem_dim, arity, 0.99]]
                                when rem_dim == None, (can beused when dont calculate dim),
                                would apply default func, with return dim self
                Note:
                the arity for accumulative_operation must be 1.
                if calculate of func rem relies on the size of ast,
                1.the size of each feature group is the same, such as n_gs.
                2.the size of ast must be the same as the size of feature group n_gs.
        special_prob: None or dict
            Examples: {"MAdd":0.5,"Self":0.5}

        Returns
        -------
        self
        """

        def change(n, pp):
            if isinstance(self_categories, dict):
                if n in special_prob:
                    pp = special_prob[n]
            return pp

        if not categories:
            categories = ["Self", "MAdd", "MSub", "MMul", "MDiv"]
        if isinstance(categories, str):
            categories = [categories, ]

        for i in categories:

            if categories_prob is "balance":
                prob1 = 1 / len([_ for _ in categories_prob if _ not in ("Self", 'Flat', "MSub", "MMul", "MDiv")])
            elif isinstance(categories_prob, float):
                prob1 = categories_prob
            else:
                raise TypeError("categories_prob accept int from (0,1] or 'balance'.")

            if i is "Self":
                p = change(i, 0.8)
                self._add_dispose(func_map_dispose()[i], arity=1, name=i, prob=p)
            elif i in ("MAdd", "MSub", "MMul", "MDiv"):
                p = change(i, 0.05)
                self._add_dispose(func_map_dispose()[i], arity=1, name=i, prob=p)
            else:
                # to be add for future
                p = change(i, prob1)
                self._add_dispose(func_map_dispose()[i], arity=1, name=i, prob=p)

        if self_categories:
            for i in self_categories:
                assert len(i) == 5, "check your input of self_categories,wihch size must be 5"
                assert i[-2] == 1, "check your input of self_categories,wihch arity must be 1"
                prob = change(i, i[4])
                self._add_dispose(sympy.Function(i[0]), arity=i[3], name=i[0], prob=prob, np_func=i[1], dim_func=i[2])

        return self

    def add_tree_to_features(self, Tree, prob=0.3):
        """

        Parameters
        ----------
        Tree: SymbolTree
        prob: int
        Returns
        -------
        self
        """
        try:
            check_array(Tree.pre_y, ensure_2d=False)
            assert Tree.y_dim is not dnan
        except(ValueError, AssertionError):
            pass
        else:
            dim = Tree.y_dim
            init_name = str(Tree)
            value = Tree.pre_y  # self.expr are not passed

            name = "new%s" % self.new_num
            self.new_num += 1
            Tree.p_name = name
            self._add_terminal(value, name, dim=dim, prob=prob, init_name=init_name)

        self.premap = self.premap.add_new()
        return self

    def add_features(self, X, y, x_dim=1, y_dim=1, prob=None, group=None,
                     feature_name=None, ):

        """

        Parameters
        ----------
        X: np.ndarray
            2D data
        y: np.ndarray
        feature_name: None, list of str
            the same size wih x.shape[1]
        x_dim: 1,list of Dim
            the same size wih x.shape[1]
        y_dim: 1,Dim
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
        if y_dim is 1:
            y_dim = dless
        self.y_dim = y_dim

        if x_dim is 1:
            x_dim = [dless for _ in range(n)]

        if prob is None:
            prob = [1 for _ in range(n)]

        if feature_name:
            assert n == len(x_dim) == len(feature_name) == len(prob)
        else:
            assert n == len(x_dim) == len(prob)

        if not group:
            group = [[]]

        for i, gi in enumerate(group):
            len_gi = len(gi)
            if len_gi > 0:
                init_name = str(["x%s" % j for j in gi])
                assert all(x_dim[gi[0]] == x_dim[i] for i in gi)
                ns = np.vstack(np.array([x_dim[i] for i in gi]))
                dim = Dim(ns)
                # dim = copy.deepcopy(x_dim[gi[0]])
                # dim.n = len_gi
                self._add_terminal(np.array(X.T[gi]),
                                   name="gx%s" % i, dim=dim, prob=prob[gi[0]],
                                   init_name=init_name
                                   )
                if feature_name:
                    fea_name = str([feature_name[j] for j in gi])
                    self.terminals_fea_map["gx%s" % i] = (init_name, fea_name)

        groups = []
        for groupi in group:
            groups.extend(groupi)

        for i, (v, dimi, probi) in enumerate(zip(X.T, x_dim, prob)):
            if i not in groups:
                self._add_terminal(v, name="x%s" % i, dim=dimi, prob=probi)
                if feature_name:
                    self.terminals_fea_map["x%s" % i] = ("x%s" % i, feature_name[i])

        self.premap = PreMap.from_shape(len(self.terminals_and_constants_repr))
        # re-generate each time.
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
            prob = [0.1 for _ in range(n)]

        assert len(c) == len(dim) == len(prob)

        for v, dimi, probi in zip(c, dim, prob):
            self._add_constant(v, name=None, dim=dimi, prob=probi)

        self.premap = PreMap.from_shape(len(self.terminals_and_constants_repr))
        # re-generate each time.
        return self

    def personal_preference(self, pers):
        """  personal_preference:list of list
        [[1,3,0.8],[2.3.0.5]]"""
        for i in pers:
            self.premap.down_others(*i)

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
    def prob_ter_con_list(self):
        return self.get_values(self.prob_ter_con, mean=True)

    @property
    def prob_pri_list(self):
        return self.get_values(self.prob_pri, mean=True)

    @property
    def prob_dispose_list(self):
        return self.get_values(self.prob_dispose, mean=True)

    @property
    def dim_ter_con_list(self):
        return self.get_values(self.dim_ter_con, mean=False)

    @property
    def primitives(self):
        return self.get_values(self.primitives_dict, mean=False)

    @property
    def dispose(self):
        return self.get_values(self.dispose_dict, mean=False)

    @property
    def terminals_and_constants(self):
        return self.get_values(self.ter_con_dict, mean=False)

    @property
    def terminals_and_constants_repr(self):
        return [sympy.Symbol(repr(i)) for i in self.terminals_and_constants]

    @property
    def data_x(self):
        return self.get_values(self.data_x_dict, mean=False)

    def compress(self):
        """Delete unnecessary detials, used before build Tree"""
        [delattr(i, "func") for i in self.dispose if hasattr(i, "func")]
        [delattr(i, "func") for i in self.primitives if hasattr(i, "func")]
        [delattr(i, "value") for i in self.terminals_and_constants if hasattr(i, "value")]
        [delattr(i, "dim") for i in self.terminals_and_constants if hasattr(i, "dim")]

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
                    string = prim.format_long(*args)
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
        return max_depth

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
        return hash(repr(self))

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
        self.p_name = None
        self.y_dim = dnan
        self.pre_y = None
        self.expr = None
        self.dim_score = 0

    def __setitem__(self, key, val):
        """keep these attribute refreshed"""
        self.p_name = None
        self.y_dim = dnan
        self.pre_y = None
        self.expr = None
        self.dim_score = 0

        _ExprTree.__setitem__(self, key, val)

    def __repr__(self):
        if self.p_name:
            return self.p_name
        else:
            return _ExprTree.__repr__(self)

    def __hash__(self):
        return hash(self.hasher(self))

    def compress(self):

        [_ExprTree.__delattr__(self, i) for i in ("coef_expr", "coef_pre_y", "coef_score", "pure_expr", "pure_pre_y")
         if hasattr(self, i)]

    def terminals(self):
        """Return terminals that occur in the expression tree."""
        return [primitive for primitive in self if primitive.arity == 0]

    def ter_site(self):
        return [i for i, primitive in enumerate(self) if primitive.arity == 0]

    def sub(self, pset):
        """
        substitute the representing name with featue_name.
        Parameters
        ----------
        pset: SymbolSet

        Returns
        -------
        """

        maps = pset.terminals_fea_map
        name_subd = str(self)
        if maps:
            for i, j1, j2 in maps.items():
                name_subd = name_subd.replace(i, j1)
                name_subd = name_subd.replace(j1, j2)
        else:
            print("Don not assign the feature_name to pset when pest.add_features")
        return name_subd

    def depart(self):
        return depart(self)

    def capsule(self):
        return ShortStr(self)

    @classmethod
    def generate(cls, pset, min_, max_, condition, per=False, *kwargs):
        """details in generate function"""
        return cls(generate(pset, min_, max_, condition, per=per, *kwargs))

    @classmethod
    def genGrow(cls, pset, min_, max_, per=False, ):
        """details in genGrow function"""
        return cls(genGrow(pset, min_, max_, per=per))

    @classmethod
    def genFull(cls, pset, min_, max_, per=False, ):
        """details in genGrow function"""
        return cls(genFull(pset, min_, max_, per=per, ))


class ShortStr:
    """short version of tree, just left name to simplify the store and transmit."""

    def __init__(self, st):
        self.reprst = repr(st)
        self.strst = str(st)

    def __str__(self):
        return self.strst

    def __repr__(self):
        return self.reprst


class CalculatePrecisionSet(SymbolSet):
    """
    Add score method to SymbolSet.
    The object can get from a worked symbolset object.
    """

    def __hash__(self):
        return hash(self.hasher(self))

    def __new__(cls, pset, scoring=None, score_pen=(1,), filter_warning=True, cal_dim=True,
                add_coef=True, inter_add=True, inner_add=False, n_jobs=1, batch_size=20, tq=True):

        cpset = super().__new__(cls)
        cpset.__dict__.update(copy.deepcopy((pset.__dict__)))

        return cpset

    def __init__(self, pset, scoring=None, score_pen=(1,), filter_warning=True, cal_dim=True,
                 add_coef=True, inter_add=True, inner_add=False, n_jobs=1, batch_size=20, tq=True):
        """

        Parameters
        ----------
        pset:SymbolSet
        scoring: Callbale, default is sklearn.metrics.r2_score
            See Also sklearn.metrics
        score_pen: tuple, default is sklearn.metrics.r2_score
            See Also sklearn.metrics
        filter_warning:bool
        score_pen: tuple of 1 or -1
            1 : best is positive, worse -np.inf
            -1 : best is negative, worse np.inf
            0 : best is positive , worse 0
        cal_dim: calculate dim or not, if not return dimless
        add_coef: bool
        inter_add: bool
        inner_add: bool
        n_jobs:int
            running core
        batch_size:int
            batch size, advice batch_size*n_jobs = inds/n
        tq:bool

        """
        _ = pset
        self.name = "CPSet"
        self.cal_dim = cal_dim
        self.score_pen = score_pen
        self.filter_warning = filter_warning
        self.scoring = scoring
        self.add_coef = add_coef
        self.inter_add = inter_add
        self.inner_add = inner_add
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.tq = tq

    def calculate_detail(self, ind):
        """

        Parameters
        ----------
        ind: SymbolTree

        Returns
        -------
        SymbolTree
        """
        ind = self.calculate_simple(ind)

        score, expr01, pre_y = calculate_score(ind.expr, self.data_x, self.y,
                                               self.terminals_and_constants_repr,
                                               add_coef=self.add_coef, inter_add=self.inter_add,
                                               inner_add=self.inner_add,
                                               scoring=self.scoring, score_pen=self.score_pen,
                                               filter_warning=self.filter_warning,
                                               np_maps=self.np_map)

        # this group should be get onetime and get all.
        ind.coef_expr = expr01
        ind.coef_pre_y = pre_y

        ind.coef_score = score

        ind.pure_expr = ind.expr
        ind.pure_pre_y = ind.pre_y

        return ind

    def calculate_simple(self, ind):
        """

        Parameters
        ----------
        ind:SymbolTree

        Returns
        -------
        SymbolTree
        """
        if isinstance(ind, SymbolTree):
            expr = compile_context(ind, self.context)
        elif isinstance(ind, sympy.Expr):
            expr = ind
        else:
            raise TypeError("must be SymbolTree or sympy.Expr")
        score, expr01, pre_y = calculate_score(expr, self.data_x, self.y,
                                               self.terminals_and_constants_repr,
                                               add_coef=False, inter_add=False, inner_add=False,
                                               scoring=self.scoring, score_pen=self.score_pen,
                                               filter_warning=self.filter_warning, np_maps=self.np_map)
        if self.cal_dim:
            dim, dim_score = calcualte_dim(expr, self.terminals_and_constants_repr,
                                           self.dim_ter_con_list, self.y_dim, self.dim_map)
        else:
            dim, dim_score = dless, 1

        ind.y_dim = dim
        ind.expr = expr01
        ind.pre_y = pre_y
        ind.dim_score = dim_score

        return ind

    def parallelize_score(self, inds):
        """

        Parameters
        ----------
        inds:SymbolTree
        Returns
        -------
        list of (score,dim,dim_score)
        """
        inds = [i.capsule() for i in inds]
        calls = functools.partial(calculate_collect, context=self.context, x=self.data_x, y=self.y,
                                  terminals_and_constants_repr=self.terminals_and_constants_repr,
                                  dim_ter_con_list=self.dim_ter_con_list, y_dim=self.y_dim,
                                  scoring=self.scoring, score_pen=self.score_pen,
                                  add_coef=self.add_coef, inter_add=self.inter_add,
                                  inner_add=self.inner_add, np_maps=self.np_map,
                                  filter_warning=self.filter_warning,
                                  dim_maps=self.dim_map, cal_dim=self.cal_dim)
        score_dim_list = parallelize(func=calls, iterable=inds, n_jobs=self.n_jobs, respective=False,
                                     tq=self.tq, batch_size=self.batch_size)

        return score_dim_list
