# -*- coding: utf-8 -*-

# @Time   : 2019/6/8 21:35
# @Author : Administrator
# @Project : feature_preparation
# @FileName: symbollearing.py
# @Software: PyCharm

"""
All part are copy from deap (https://github.com/DEAP/deap)
"""
import array
import copy
import functools
import random
import sys
import warnings
from collections import Sequence, defaultdict
from copy import deepcopy
from functools import partial
from operator import mul, truediv

import numpy as np
import pandas as pd
import sympy
from deap import gp
from deap.algorithms import varAnd
from deap.gp import PrimitiveSet
from deap.tools import Logbook
from scipy import optimize
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import explained_variance_score, make_scorer
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import assert_all_finite, check_array

from ..tools.exports import Store
from ..tools.tool import check_random_state, parallize

warnings.filterwarnings("ignore")


class Toolbox(object):

    def __init__(self):

        self.register("clone", deepcopy)

        self.register("map", map)

    def register(self, alias, function, *args, **kargs):

        pfunc = partial(function, *args, **kargs)

        pfunc.__name__ = alias

        pfunc.__doc__ = function.__doc__

        if hasattr(function, "__dict__") and not isinstance(function, type):
            # Some functions don't have a dictionary, in these cases

            # simply don't copy it. Moreover, if the function is actually

            # a class, we do not want to copy the dictionary.

            pfunc.__dict__.update(function.__dict__.copy())

        setattr(self, alias, pfunc)

    def unregister(self, alias):

        """Unregister *alias* from the toolbox.



        :param alias: The name of the operator to remove from the toolbox.

        """

        delattr(self, alias)

    def decorate(self, alias, *decorators):

        """Decorate *alias* with the specified *decorators*, *alias*

        has to be a registered function in the current toolbox.



        :param alias: The name of the operator to decorate.

        :param decorators: One or more function decorator. If multiple

                          decorators are provided they will be applied in

                          order, with the last decorator decorating all the

                          others.



        .. note::

            Decorate a function using the toolbox makes it unpicklable, and

            will produce an error on pickling. Although this limitation is not

            relevant in most cases, it may have an impact on distributed

            environments like multiprocessing.

            A function can still be decorated manually before it is added to

            the toolbox (using the @ notation) in order to be picklable.

        """

        pfunc = getattr(self, alias)

        function, args, kargs = pfunc.func, pfunc.args, pfunc.keywords

        for decorator in decorators:
            function = decorator(function)

        self.register(alias, function, *args, **kargs)


class Fitness(object):
    """The fitness is a measure of quality of a solution. If *values* are

    provided as a tuple, the fitness is initalized using those values,

    otherwise it is empty (or invalid).



    :param values: The initial values of the fitness as a tuple, optional.



    Fitnesses may be compared using the ``>``, ``<``, ``>=``, ``<=``, ``==``,

    ``!=``. The comparison of those operators is made lexicographically.

    Maximization and minimization are taken care off by a multiplication

    between the :attr:`weights` and the fitness :attr:`values`. The comparison

    can be made between fitnesses of different size, if the fitnesses are

    equal until the extra elements, the longer fitness will be superior to the

    shorter.



    Different types of fitnesses are created in the :ref:`creating-types`

    tutorial.



    .. note::

       When comparing fitness values that are **minimized**, ``a > b`` will

       return :data:`True` if *a* is **smaller** than *b*.

    """

    weights = None

    """The weights are used in the fitness comparison. They are shared among

    all fitnesses of the same type. When subclassing :class:`Fitness`, the

    weights must be defined as a tuple where each element is associated to an

    objective. A negative weight element corresponds to the minimization of

    the associated objective and positive weight to the maximization.



    .. note::

        If weights is not defined during subclassing, the following error will

        occur at instantiation of a subclass fitness object:



        ``TypeError: Can't instantiate abstract <class Fitness[...]> with

        abstract attribute weights.``

    """

    wvalues = ()

    """Contains the weighted values of the fitness, the multiplication with the

    weights is made when the values are set via the property :attr:`values`.

    Multiplication is made on setting of the values for efficiency.



    Generally it is unnecessary to manipulate wvalues as it is an internal

    attribute of the fitness used in the comparison operators.

    """

    def __init__(self, values=()):

        if self.weights is None:
            raise TypeError("Can't instantiate abstract %r with abstract "

                            "attribute weights." % self.__class__)

        if not isinstance(self.weights, Sequence):
            raise TypeError("Attribute weights of %r must be a sequence."

                            % self.__class__)

        if len(values) > 0:
            self.values = values

    def getValues(self):

        return tuple(map(truediv, self.wvalues, self.weights))

    def setValues(self, values):

        try:

            self.wvalues = tuple(map(mul, values, self.weights))

        except TypeError:

            _, _, traceback = sys.exc_info()

            raise TypeError("Both weights and assigned values must be a "

                            "sequence of numbers when assigning to values of "

                            "%r. Currently assigning value(s) %r of %r to a "

                            "fitness with weights %s."

                            % (self.__class__, values, type(values),

                               self.weights)).with_traceback(traceback)

    def delValues(self):

        self.wvalues = ()

    values = property(getValues, setValues, delValues,

                      ("Fitness values. Use directly ``individual.fitness.values = values`` "

                       "in order to set the fitness and ``del individual.fitness.values`` "

                       "in order to clear (invalidate) the fitness. The (unweighted) fitness "

                       "can be directly accessed via ``individual.fitness.values``."))

    def dominates(self, other, obj=slice(None)):

        not_equal = False

        for self_wvalue, other_wvalue in zip(self.wvalues[obj], other.wvalues[obj]):

            if self_wvalue > other_wvalue:

                not_equal = True

            elif self_wvalue < other_wvalue:

                return False

        return not_equal

    @property
    def valid(self):

        """Assess if a fitness is valid or not."""

        return len(self.wvalues) != 0

    def __hash__(self):

        return hash(self.wvalues)

    def __gt__(self, other):

        return not self.__le__(other)

    def __ge__(self, other):

        return not self.__lt__(other)

    def __le__(self, other):

        return self.wvalues <= other.wvalues

    def __lt__(self, other):

        return self.wvalues < other.wvalues

    def __eq__(self, other):

        return self.wvalues == other.wvalues

    def __ne__(self, other):

        return not self.__eq__(other)

    def __deepcopy__(self, memo):

        """Replace the basic deepcopy function with a faster one.



        It assumes that the elements in the :attr:`values` tuple are

        immutable and the fitness does not contain any other object

        than :attr:`values` and :attr:`weights`.

        """

        copy_ = self.__class__()

        copy_.wvalues = self.wvalues

        return copy_

    def __str__(self):

        """Return the values of the Fitness object."""

        return str(self.values if self.valid else tuple())

    def __repr__(self):

        """Return the Python code to build a copy of the object."""

        return "%s.%s(%r)" % (self.__module__, self.__class__.__name__,

                              self.values if self.valid else tuple())


class_replacers = {}


class _numpy_array(np.ndarray):
    def __deepcopy__(self, memo, **kwargs):
        """Overrides the deepcopy from numpy.ndarray that does not copy
        the object's attributes. This one will deepcopy the array and its
        :param **kwargs:
        :attr:`__dict__` attribute.
        """
        copy_ = np.ndarray.copy(self)
        copy_.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return copy_

    @staticmethod
    def __new__(cls, iterable):
        """Creates a new instance of a numpy.ndarray from a function call.
        Adds the possibility to instanciate from an iterable."""
        return np.array(list(iterable)).view(cls)

    def __setstate__(self, state, **kwargs):
        self.__dict__.update(state)

    def __reduce__(self):
        return self.__class__, (list(self),), self.__dict__


class_replacers[np.ndarray] = _numpy_array


class _array(array.array):
    @staticmethod
    def __new__(cls, seq=()):
        return super(_array, cls).__new__(cls, cls.typecode, seq)

    def __deepcopy__(self, memo):
        """Overrides the deepcopy from array.array that does not copy
        the object's attributes and class type.
        """
        cls = self.__class__
        copy_ = cls.__new__(cls, self)
        memo[id(self)] = copy_
        copy_.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return copy_

    def __reduce__(self):
        return self.__class__, (list(self),), self.__dict__


class_replacers[array.array] = _array


def create(name, base, **kargs):
    if name in globals():
        warnings.warn("A class named '{0}' has already been created and it "
                      "will be overwritten. Consider deleting previous "
                      "creation of that class or rename it.".format(name),
                      RuntimeWarning)

    dict_inst = {}
    dict_cls = {}
    for obj_name, obj in kargs.items():
        if isinstance(obj, type):
            dict_inst[obj_name] = obj
        else:
            dict_cls[obj_name] = obj

    # Check if the base class has to be replaced
    if base in class_replacers:
        base = class_replacers[base]

    # A DeprecationWarning is raised when the object inherits from the
    # class "object" which leave the option of passing arguments, but
    # raise a warning stating that it will eventually stop permitting
    # this option. Usually this happens when the base class does not
    # override the __init__ method from object.
    def initType(self, *args, **kargs):
        """Replace the __init__ function of the new type, in order to
        add attributes that were defined with **kargs to the instance.
        """
        for obj_name, obj in dict_inst.items():
            setattr(self, obj_name, obj())
        if base.__init__ is not object.__init__:
            base.__init__(self, *args, **kargs)

    objtype = type(str(name), (base,), dict_cls)
    objtype.__init__ = initType
    return objtype


class FixedTerminal(object):
    __slots__ = ('name', 'value', 'conv_fct', 'arity')

    def __init__(self, terminal):
        self.value = terminal
        self.name = str(terminal)
        self.conv_fct = str
        self.arity = 0

    def format(self):
        return self.conv_fct(self.value)

    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot) for slot in self.__slots__)
        else:
            return NotImplemented

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return self.name

    __repr__ = __str__


class FixedPrimitive(object):
    __slots__ = ('name', 'arity', 'args', 'seq')

    def __init__(self, name, arity):
        self.name = name
        self.arity = arity
        self.args = []
        args = ", ".join(map("{{{0}}}".format, list(range(self.arity))))
        self.seq = "{name}({args})".format(name=self.name, args=args)

    def format(self, *args):
        return self.seq.format(*args)

    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot) for slot in self.__slots__)
        else:
            return NotImplemented

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return self.name

    __repr__ = __str__


class FixedPrimitiveSet(object):

    def __init__(self, name):
        self.terminals = []
        self.primitives = []
        self.terms_count = 0
        self.prims_count = 0
        self.arguments = []
        self.context = {"__builtins__": None}
        self.dimtext = {"__builtins__": None}
        self.mapping = dict()
        self.name = name

    def addPrimitive(self, primitive, arity, name=None):

        if name is None:
            name = primitive.__name__

        prim = FixedPrimitive(name, arity)

        assert name not in self.context, "Primitives are required to have a unique name. " \
                                         "Consider using the argument 'name' to rename your " \
                                         "second '%s' primitive." % (name,)

        self.primitives.append(prim)
        self.context[prim.name] = primitive
        self.prims_count += 1

    def addTerminal(self, terminal, name=None):

        if name is None and callable(terminal):
            name = str(terminal)

        assert name not in self.context, "Terminals are required to have a unique name. " \
                                         "Consider using the argument 'name' to rename your " \
                                         "second %s terminal." % (name,)

        if name is not None:
            self.context[name] = terminal

        prim = FixedTerminal(terminal)
        self.terminals.append(prim)
        self.terms_count += 1

    @property
    def terminalRatio(self):
        """Return the ratio of the number of terminals on the number of all
        kind of primitives.
        """
        return self.terms_count / float(self.terms_count + self.prims_count)


def sympyPrimitiveSet(rep_name, types="Fixed", categories=("Add", "Mul", "Abs", "exp"), power_categories=None,
                      partial_categories=None, self_categories=None, name=None, dim=None,
                      definate_operate=None, definate_variable=None, linkage=None, ):
    """
        :type partial_categories: double list
        partial_categories = [["Add","Mul"],["x4"]]
        :type power_categories: list
        index_categories=[0.5,1,2,3]
        :type dim: list,tuple
        :type name: list,tuple
        :type rep_name: list,tuple
        :type categories: list,tuple
        :param self_categories:
        def rem(a):
            return 1-a
        self_categories = [[rem, 1, 'rem']]
        :type linkage: list,tuple
        linkage = [[]]
        :type definate_variable: list,tuple
        definate_variable = [(-1, [1, ]), ]
        :type definate_operate: list,tuple
        definate_operate = [(-1, [0, ]), ]
        :param types
    """

    def Div(left, right):
        return left / right

    def Sub(left, right):
        return left - right

    def zeroo(_):
        return 0

    def oneo(_):
        return 1

    def rem(a):
        return 1 - a

    def self(_):
        return _

    functions2 = {"Add": sympy.Add, 'Sub': Sub, 'Mul': sympy.Mul, 'Div': Div, 'Max': sympy.Max, "Min": sympy.Min}
    functions1 = {"sin": sympy.sin, 'cos': sympy.cos, 'exp': sympy.exp, 'log': sympy.ln,
                  'Abs': sympy.Abs, "Neg": functools.partial(sympy.Mul, -1.0),
                  "Rec": functools.partial(sympy.Pow, e=-1.0),
                  'Zeroo': zeroo, "Oneo": oneo, "Rem": rem, "Self": self}

    pset0 = FixedPrimitiveSet('main') if types in ["fix", "fixed", "Fix", "Fixed"] else PrimitiveSet('main', 0)

    for i in categories:
        if i in functions2:
            pset0.addPrimitive(functions2[i], arity=2, name=i)
        if i in functions1:
            pset0.addPrimitive(functions1[i], arity=1, name=i)

    if power_categories:
        for j, i in enumerate(power_categories):
            pset0.addPrimitive(functools.partial(sympy.Pow, e=i), arity=1, name='pow%s' % j)

    if partial_categories:
        for partial_categoriesi in partial_categories:
            for i in partial_categoriesi[0]:
                for j in partial_categoriesi[1]:
                    if i in ["Mul", "Add"]:
                        pset0.addPrimitive(functools.partial(functions2[i], sympy.Symbol(j)), arity=1,
                                           name="{}_{}".format(i, j))
                    else:
                        pset0.addPrimitive(functools.partial(functions2[i], right=sympy.Symbol(j)), arity=1,
                                           name="{}_{}".format(i, j))
    if self_categories:
        for i in self_categories:
            pset0.addPrimitive(i[0], i[1], i[2])

    # define terminal
    if isinstance(rep_name[0], str):
        rep_name = [sympy.Symbol(i) for i in rep_name]
    if dim is None:
        dim = [1] * len(rep_name)
    if name is None:
        name = rep_name

    assert len(dim) == len(name) == len(rep_name)

    for sym in rep_name:
        pset0.addTerminal(sym, name=str(sym))

    # define limit
    if linkage is None:
        linkage = [[]]
    assert isinstance(linkage, (list, tuple))
    if not isinstance(linkage[0], (list, tuple)):
        linkage = [linkage, ]

    if types in ["fix", "fixed", "Fix", "Fixed"]:
        dict_pri = dict(zip([_.name for _ in pset0.primitives], range(len(pset0.primitives))))
        dict_ter = dict(zip([_.name for _ in pset0.terminals], range(len(pset0.terminals))))
    else:
        dict_pri = dict(zip([_.name for _ in pset0.primitives[object]], range(len(pset0.primitives))))
        dict_ter = dict(zip([_.name for _ in pset0.terminals[object]], range(len(pset0.terminals))))

    if definate_operate:
        definate_operate = [list(i) for i in definate_operate]
        for i, j in enumerate(definate_operate):
            j = list(j)
            definate_operate[i][1] = [dict_pri[_] if _ in dict_pri else _ for _ in j[1]]
    if definate_variable:
        definate_variable = [list(i) for i in definate_variable]
        for i, j in enumerate(definate_variable):
            j = list(j)
            definate_variable[i][1] = [dict_ter[_] if _ in dict_ter else _ for _ in j[1]]

    pset0.definate_operate = definate_operate
    pset0.definate_variable = definate_variable
    pset0.linkage = linkage
    pset0.rep_name_list = rep_name
    pset0.name_list = name
    pset0.dim_list = dim

    return pset0


def compile_(expr_, pset):
    code = str(expr_)
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


def sub(expr01, subed, subs):
    listt = list(zip(subed, subs))
    return expr01.subs(listt)


def add_coefficient(expr01, inter_add=None, iner_add=None, random_add=None):
    def get_args(expr_):
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
    a_list = []
    #
    if isinstance(expr01, sympy.Add):
        for i, j in enumerate(expr01.args):
            Wi = sympy.Symbol("W%s" % i)
            expr01 = expr01.subs(j, Wi * j)
            a_list.append(Wi)
    else:
        A = sympy.Symbol("A")
        expr01 = expr01 * A
        a_list.append(A)

    if inter_add:
        B = sympy.Symbol("B")
        expr01 = expr01 + B
        a_list.append(B)

    if iner_add:
        cho_add = [i.args for i in arg_list if isinstance(i, sympy.Add)]
        [cho.extend(i) for i in cho_add]

    if random_add:
        random_state = check_random_state(3)
        lest = [i for i in arg_list if i not in cho]
        if len(lest) != 0:
            cho2 = random_state.sample(lest, 1)
            cho.extend(cho2)
    # #
    a_cho = [sympy.Symbol("k%s" % i) for i in range(len(cho))]
    for ai, choi in zip(a_cho, cho):
        expr01 = expr01.subs(choi, ai * choi)
    a_list.extend(a_cho)

    return expr01, a_list


def my_custom_loss_func(y_true, y_pred):
    diff = - np.abs(y_true - y_pred) / y_true + 1
    return np.mean(diff)


mre_score = make_scorer(my_custom_loss_func, greater_is_better=True)


def calculate(individual, pset, x, y, score_method=explained_variance_score, add_coeff=True, filter_warning=True,
              **kargs):
    # '''1 not expand'''
    expr_no = sympy.sympify(compile_(individual, pset))
    # '''2 expand by sympy.expand,long expr is slow, use if when needed'''
    # expr_no = sympy.expand(compile_(individual, pset), deep=False, power_base=False, power_exp=False, mul=True,
    #                        log=False, multinomial=False)
    # '''3 expand specially '''
    if isinstance(expr_no, sympy.Mul) and len(expr_no.args) == 2:
        if isinstance(expr_no.args[0], sympy.Add) and expr_no.args[1].args == ():
            expr_no = sum([i * expr_no.args[1] for i in expr_no.args[0].args])
        if isinstance(expr_no.args[1], sympy.Add) and expr_no.args[0].args == ():
            expr_no = sum([i * expr_no.args[0] for i in expr_no.args[1].args])
        else:
            pass

    score, expr = calculate_expr(expr_no, pset, x, y, score_method=score_method, add_coeff=add_coeff,
                                 filter_warning=filter_warning, **kargs)
    return score, expr


def calculate_expr(expr01, pset, x, y, score_method=explained_variance_score, add_coeff=True,
                   del_no_important=False, filter_warning=True, **kargs):
    terminals = pset.terminals[object] if isinstance(pset.terminals, defaultdict) else pset.terminals
    if filter_warning:
        warnings.filterwarnings("ignore")

    expr00 = deepcopy(expr01)

    if not score_method:
        score_method = r2_score
    if add_coeff:
        expr01, a_list = add_coefficient(expr01, **kargs)
        try:
            func0 = sympy.utilities.lambdify([_.value for _ in terminals] + a_list, expr01)

            def func(x_, p):
                num_list = []
                num_list.extend([*x_.T])
                num_list.extend(p)
                return func0(*num_list)

            def res(p, x_, y_):
                return y_ - func(x_, p)

            result = optimize.least_squares(res, x0=[1] * len(a_list), args=(x, y), loss='huber', ftol=1e-4)

            cof = result.x
            cof_ = []
            for a_listi, cofi in zip(a_list, cof):
                if "A" or "W" in a_listi.name:
                    cof_.append(cofi)
                else:
                    cof_.append(np.round(cofi, decimals=3))
            cof = cof_
            for ai, choi in zip(a_list, cof):
                expr01 = expr01.subs(ai, choi)
        except (ValueError, NameError, TypeError):
            expr01 = expr00
    else:
        pass
    try:
        func0 = sympy.utilities.lambdify([_.value for _ in terminals], expr01)
        re = func0(*x.T)
        assert_all_finite(re)
        check_array(re, ensure_2d=False)

    except (ValueError, DataConversionWarning, TypeError, NameError):
        score = -0
    else:
        def cv_expr(expr_):
            func0_ = sympy.utilities.lambdify([_.value for _ in terminals], expr_)
            ss = ShuffleSplit(n_splits=5, test_size=0.20, random_state=10)
            score_ = []
            for train_index, _ in ss.split(x, y):
                x_train = x[train_index]
                y_train = y[train_index]
                y_train_pre = check_array(func0_(*x_train.T), ensure_2d=False)
                score_.append(score_method(y_train, y_train_pre))
            score_ = np.mean(score_)
            return score_

        if del_no_important and isinstance(expr01, sympy.Add) and len(expr01.args) >= 3:
            re_list = []
            for expri in expr01.args:
                if not isinstance(expri, sympy.Float):
                    func0 = sympy.utilities.lambdify([_.value for _ in terminals], expri)
                    re = np.mean(func0(*x.T))
                    if abs(re) > abs(0.001 * np.mean(y)):
                        re_list.append(expri)
                else:
                    re_list.append(expri)
            if len(re_list) <= 1:
                expr01 = sum(re_list)
                score = -0
            else:
                expr01 = sum(re_list)
                score = cv_expr(expr01)
        else:
            score = cv_expr(expr01)

    return score, expr01


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, pset=None, store=True, alpha=1):
    logbook = Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    fitnesses = parallize(n_jobs=4, func=toolbox.evaluate, iterable=invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit[0],
        ind.expr = fit[1]

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    data_all = {}
    # Begin the generational process
    for gen in range(1, ngen + 1):

        if store:
            if pset:
                subp = partial(sub, subed=pset.rep_name_list, subs=pset.name_list)
                data = [{"score": i.fitness.values[0], "expr": subp(i.expr)} for i in halloffame.items[-5:]]
            else:
                data = [{"score": i.fitness.values[0], "expr": i.expr} for i in halloffame.items[-5:]]
            data_all['gen%s' % gen] = data

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        if halloffame is not None:
            offspring.extend(halloffame)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        fitnesses = parallize(n_jobs=4, func=toolbox.evaluate, iterable=invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0],
            ind.expr = fit[1]
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

            if halloffame.items[-1].fitness.values[0] >= 0.95:
                print(halloffame.items[-1])
                print(halloffame.items[-1].fitness.values[0])
                break

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
    store = Store()
    store.to_txt(data_all)
    return population, logbook


def multiEaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
                  halloffame=None, verbose=__debug__, pset=None, store=True, alpha=1):
    logbook = Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    # fitnesses = list(toolbox.map(toolbox.evaluate, [str(_) for _ in invalid_ind]))
    # fitnesses2 = toolbox.map(toolbox.evaluate2, [str(_) for _ in invalid_ind])
    fitnesses = parallize(n_jobs=6, func=toolbox.evaluate, iterable=[str(_) for _ in invalid_ind])
    fitnesses2 = parallize(n_jobs=6, func=toolbox.evaluate2, iterable=[str(_) for _ in invalid_ind])

    def funcc(a, b):

        return (alpha * a + b) / 2

    for ind, fit, fit2 in zip(invalid_ind, fitnesses, fitnesses2):
        ind.fitness.values = funcc(fit[0], fit2[0]),
        ind.values = (fit[0], fit2[0])
        ind.expr = (fit[1], fit2[1])

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    data_all = {}
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        if halloffame is not None:
            offspring.extend(halloffame.items[-2:])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # fitnesses = toolbox.map(toolbox.evaluate, [str(_) for _ in invalid_ind])
        # fitnesses2 = toolbox.map(toolbox.evaluate2, [str(_) for _ in invalid_ind])
        fitnesses = parallize(n_jobs=6, func=toolbox.evaluate, iterable=[str(_) for _ in invalid_ind])
        fitnesses2 = parallize(n_jobs=6, func=toolbox.evaluate2, iterable=[str(_) for _ in invalid_ind])

        for ind, fit, fit2 in zip(invalid_ind, fitnesses, fitnesses2):
            ind.fitness.values = funcc(fit[0], fit2[0]),
            ind.values = (fit[0], fit2[0])
            ind.expr = (fit[1], fit2[1])

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
            if halloffame.items[-1].fitness.values[0] >= 0.95:
                print(halloffame.items[-1])
                print(halloffame.items[-1].fitness.values[0])
                print(halloffame.items[-1].values[0])
                print(halloffame.items[-1].values[1])
                break

        if store:
            if pset:
                subp = partial(sub, subed=pset.rep_name_list, subs=pset.name_list)
                data = [{"score": i.values[0], "expr": subp(i.expr[0])} for i in halloffame.items[-2:]]
                data2 = [{"score": i.values[1], "expr": subp(i.expr[1])} for i in halloffame.items[-2:]]
            else:
                data = [{"score": i.values[0], "expr": i.expr} for i in halloffame.items[-2:]]
                data2 = [{"score": i.values[1], "expr": i.expr[2]} for i in halloffame.items[-2:]]
            data_all['gen%s' % gen] = list(zip(data, data2))

        # Replace the current population by the offspring
        population[:] = offspring
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
    if store:
        store1 = Store()
        store1.to_txt(data_all)

    return population, logbook


def get_name(x):
    if isinstance(x, pd.DataFrame):
        name = x.columns.values
        name = [sympy.Symbol(i) for i in name]
        rep_name = [sympy.Symbol("x%d" % i) for i in range(len(name))]

    elif isinstance(x, np.ndarray):
        check_array(x)
        name = x.shape[1]
        name = [sympy.Symbol("x%d" % i) for i in range(name)]
        rep_name = [sympy.Symbol("x%d" % i) for i in range(len(name))]
    else:
        raise TypeError("just support np.ndarray and pd.DataFrame")

    return name, rep_name


class FixedExpressionTree(list):
    hasher = str

    def __init__(self, content):
        list.__init__(self, content)

        assert sum(_.arity - 1 for _ in self.primitives) + 1 >= len(self.terminals)
        assert len(self.terminals) >= 2

    @property
    def root(self):
        len_ter = len(self.terminals) - 1
        num_pri = list(range(len(self.primitives)))
        num_pri.reverse()
        i = 0
        for i in num_pri:
            if len_ter == 0:
                break
            elif len_ter <= 0:
                raise ("Add terminals or move back the {}".format(self[i - 1]),
                       "because the {} have insufficient terminals, need {},but get {}".format(self[i - 1],
                                                                                               self[i - 1].arity,
                                                                                               len_ter - self[
                                                                                                   i - 1].arity)
                       )
            len_ter = len_ter - self[i].arity + 1
        # return i  # for wencheng
        return i + 1

    def __deepcopy__(self, memo):

        new = self.__class__(self)
        new.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return new

    def __hash__(self):
        return hash(self.hasher(self))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return self.pri_node_str(self.root)

    __repr__ = __str__

    @property
    def pri_site(self):
        return tuple([p for p, primitive in enumerate(self) if primitive.arity >= 1 and p >= self.root])

    @property
    def ter_site(self):
        return tuple([t for t, primitive in enumerate(self) if primitive.arity == 0])

    @property
    def primitives(self):
        """Return primitives that occur in the expression tree."""
        return [primitive for primitive in self if primitive.arity >= 1]

    @property
    def terminals(self):
        """Return terminals that occur in the expression tree."""
        return [primitive for primitive in self if primitive.arity == 0]

    # @property
    # def str_branch(self):
    #     def cal():
    #         coup_dict = {}
    #         coup = []
    #         for _ in range(pri_i.arity):
    #             coup.append(terminals.pop())
    #         coup.reverse()
    #         coup_dict[pri_i] = coup
    #         terminals_new.append(coup_dict)
    #
    #     terminals = self.terminals
    #     primitives = self.primitives
    #     primitives.reverse()
    #     terminals_new = []
    #     for i, pri_i in enumerate(primitives):
    #         if len(terminals) >= pri_i.arity:
    #             cal()
    #         else:
    #             terminals_new.reverse()
    #             terminals.extend(terminals_new)
    #             terminals_new = []
    #             if len(terminals) >= pri_i.arity:
    #                 cal()
    #             else:
    #                 break
    #
    #     result = terminals_new or terminals
    #     return result[0]

    @property
    def number_branch(self):
        def cal():
            coup_dict = {}
            coup = []
            for _ in range(pri_i.arity):
                coup.append(terminals.pop())
            coup.reverse()
            coup_dict[len_pri - i - 1] = coup
            terminals_new.append(coup_dict)

        primitives = self.primitives
        primitives.reverse()
        len_pri = len(primitives)
        terminals = list(range(len_pri, len(self.terminals) + len_pri))
        terminals_new = []
        for i, pri_i in enumerate(primitives):
            if len(terminals) >= pri_i.arity:
                cal()
            else:
                terminals_new.reverse()
                terminals.extend(terminals_new)
                terminals_new = []
                if len(terminals) >= pri_i.arity:
                    cal()
                else:
                    break
        result = terminals_new or terminals
        return result[0]

    @classmethod
    def search_y_name(cls, name):
        list_name = []
        for i in range(len(cls)):
            if cls[i].name == name:
                list_name.append(i)
        return list_name

    def number_branch_index(self, index):
        if index < self.root or index > len(self.primitives):
            raise IndexError("not a primitives index")
        else:
            def run_index(number_branch=None):
                if number_branch is None:
                    number_branch = self.number_branch
                jk = list(number_branch.keys())[0]
                ji = list(number_branch.values())[0]
                if jk == index:
                    return number_branch
                else:
                    repr1 = []
                    for jii in ji:
                        if isinstance(jii, dict):
                            repr1 = run_index(jii)
                        else:
                            repr1 = []
                        if repr1:
                            break
                    return repr1
        set1 = run_index()
        # set1.sort()
        return set1

    def str_run(self, number_branch=None):
        if number_branch is None:
            number_branch = self.number_branch
        # print(number_branch)
        args = []
        jk = list(number_branch.keys())[0]
        ji = list(number_branch.values())[0]

        for jii in ji:
            if isinstance(jii, dict):
                repr1 = self.str_run(jii)
            else:
                repr1 = self[jii].name
            args.append(repr1)
        repr1 = self[jk].format(*args)

        return repr1

    def pri_node_str(self, i):
        return self.str_run(self.number_branch_index(i))

    def pri_node_index(self, i):

        def run_index(number_branch=None):
            if number_branch is None:
                number_branch = self.number_branch
            jk = list(number_branch.keys())[0]
            ji = list(number_branch.values())[0]
            sub_index = []
            for jii in ji:
                if isinstance(jii, dict):
                    repr1 = run_index(jii)
                else:
                    repr1 = [jii]
                sub_index.extend(repr1)
            sub_index.append(jk)

            return sub_index

        res = run_index(number_branch=self.number_branch_index(i))
        res = list(set(res))
        res.sort()
        return res


class ExpressionTree(gp.PrimitiveTree):
    hasher = str

    def __init__(self, content):
        super(ExpressionTree, self).__init__(content)

    def __repr__(self):
        """Symbolic representation of the expression tree."""
        repr1 = ''
        stack = []
        for node in self:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                repr1 = prim.format(*args)
                if len(stack) == 0:
                    break
                stack[-1][1].append(repr1)
        return repr1

    def __hash__(self):
        return hash(self.hasher(self))

    def __eq__(self, other):
        return hash(self) == hash(other)

    @property
    def terminals(self):
        """Return terminals that occur in the expression tree."""
        return [primitive for primitive in self if primitive.arity == 0]

    @property
    def pri_site(self):
        return [i for i, primitive in enumerate(self) if primitive.arity >= 1]

    @property
    def ter_site(self):
        return [i for i, primitive in enumerate(self) if primitive.arity == 0]

    @property
    def primitives(self):
        """Return primitives that occur in the expression tree."""
        return [primitive for primitive in self if primitive.arity >= 1]


def cxOnePoint_index(ind1, ind2, pset):
    linkage = pset.linkage
    root = max(ind1.root, ind2.root)
    index = random.randint(root, root + len(ind1.pri_site))
    ind10 = copy.copy(ind1)
    ind20 = copy.copy(ind2)
    ind10[index:] = ind2[index:]
    ind20[index:] = ind1[index:]
    if linkage:
        for i in linkage:
            for _ in i:
                ind10[_] = ind10[i[0]]
                ind20[_] = ind20[i[0]]
    return ind10, ind20


def mutUniForm_index(ind1, pset, ):
    ind10 = copy.copy(ind1)
    linkage = pset.linkage
    pri2 = [i for i in pset.primitives if i.arity == 2]
    pri1 = [i for i in pset.primitives if i.arity == 1]
    index = random.choice(ind10.pri_site)
    ind10[index] = random.choice(pri2) if random.random() > 0.5 * len(pri1) / len(
        pset.primitives) else random.choice(pri1)

    definate_operate = pset.definate_operate

    if definate_operate:
        for i in definate_operate:
            ind10[ind1.pri_site[i[0]]] = pset.primitives[random.choice(i[1])]

    if linkage:
        for i in linkage:
            for _ in i:
                ind10[_] = ind10[i[0]]
    return ind10,


def produce(container, generator):
    return container(generator())


def generate(pset, min_=None, max_=None):
    if max_ is None:
        max_ = len(pset.terminals)
    if min_ is None:
        min_ = max_

    pri2 = [i for i in pset.primitives if i.arity == 2]
    pri1 = [i for i in pset.primitives if i.arity == 1]

    max_varibale_set_long = max_
    varibale_set_long = random.randint(min_, max_varibale_set_long)
    '''random'''
    # trem_set = random.sample(pset.terminals, varibale_set_long) * 20
    '''sequence'''
    trem_set = pset.terminals[:varibale_set_long] * 5
    init_operator_long = max_varibale_set_long * 3

    individual1 = []
    for i in range(init_operator_long):
        trem = random.choice(pri2) if random.random() > 0.5 * len(pri1) / len(
            pset.primitives) else random.choice(pri1)
        individual1.append(trem)
    individual2 = []
    for i in range(varibale_set_long):
        trem = trem_set[i]
        individual2.append(trem)
    # define protect primitives
    pri2 = [i for i in pset.primitives if i.arity == 2]
    protect_individual = []
    for i in range(varibale_set_long):
        trem = random.choice(pri2)
        protect_individual.append(trem)

    definate_operate = pset.definate_operate
    definate_variable = pset.definate_variable
    linkage = pset.linkage

    if definate_operate:
        for i in definate_operate:
            individual1[i[0]] = pset.primitives[random.choice(i[1])]

    if definate_variable:
        for i in definate_variable:
            individual2[i[0]] = pset.terminals[random.choice(i[1])]

    individual_all = protect_individual + individual1 + individual2
    if linkage:
        for i in linkage:
            for _ in i:
                individual_all[_] = individual_all[i[0]]

    return individual_all
