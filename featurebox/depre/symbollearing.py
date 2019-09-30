# -*- coding: utf-8 -*-

# @Time   : 2019/6/8 21:35
# @Author : Administrator
# @Project : feature_preparation
# @FileName: symbollearing.py
# @Software: PyCharm

import array
import copy
import functools
import sys
import warnings
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from functools import partial
from itertools import chain
from operator import attrgetter, mul, truediv, eq

import numpy as np
import pandas as pd
import sympy
from scipy import optimize
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import r2_score, explained_variance_score, make_scorer
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import assert_all_finite, check_array
from feature_toolbox.tools.data_export import Store
from feature_toolbox.tools.tool import check_Random_state, parallize, time_this_function

warnings.filterwarnings("ignore")

"""
All part are copy from deap (https://github.com/DEAP/deap)"""
"""
tool
"""

"""
All part are copy from deap (https://github.com/DEAP/deap)
Just for escape python version warning
"""


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


def identity(obj):
    """Returns directly the argument *obj*.

    """

    return obj


class History(object):

    def __init__(self):

        self.genealogy_index = 0

        self.genealogy_history = dict()

        self.genealogy_tree = dict()

    def update(self, individuals):

        try:

            parent_indices = tuple(ind.history_index for ind in individuals)

        except AttributeError:

            parent_indices = tuple()

        for ind in individuals:
            self.genealogy_index += 1

            ind.history_index = self.genealogy_index

            self.genealogy_history[self.genealogy_index] = deepcopy(ind)

            self.genealogy_tree[self.genealogy_index] = parent_indices

    @property
    def decorator(self):

        """Property that returns an appropriate decorator to enhance the

        operators of the toolbox. The returned decorator assumes that the

        individuals are returned by the operator. First the decorator calls

        the underlying operation and then calls the :func:`update` function

        with what has been returned by the operator. Finally, it returns the

        individuals with their history parameters modified according to the

        update function.

        """

        def decFunc(func):
            def wrapFunc(*args, **kargs):
                individuals = func(*args, **kargs)

                self.update(individuals)

                return individuals

            return wrapFunc

        return decFunc

    def getGenealogy(self, individual, max_depth=float("inf")):

        """Provide the genealogy tree of an *individual*. The individual must

        have an attribute :attr:`history_index` as defined by

        :func:`~deap.tools.History.update` in order to retrieve its associated

        genealogy tree. The returned graph contains the parents up to

        *max_depth* variations before this individual. If not provided

        the maximum depth is up to the begining of the evolution.



        :param individual: The individual at the root of the genealogy tree.

        :param max_depth: The approximate maximum distance between the root

                          (individual) and the leaves (parents), optional.

        :returns: A dictionary where each key is an individual index and the

                  values are a tuple corresponding to the index of the parents.

        """

        gtree = {}

        visited = set()  # Adds memory to the breadth first search

        def genealogy(index, depth):

            if index not in self.genealogy_tree:
                return

            depth += 1

            if depth > max_depth:
                return

            parent_indices = self.genealogy_tree[index]

            gtree[index] = parent_indices

            for ind in parent_indices:

                if ind not in visited:
                    genealogy(ind, depth)

                visited.add(ind)

        genealogy(individual.history_index, 0)

        return gtree


class Statistics(object):

    def __init__(self, key=identity):
        self.key = key

        self.functions = dict()

        self.fields = []

    def register(self, name, function, *args, **kargs):
        self.functions[name] = partial(function, *args, **kargs)

        self.fields.append(name)

    def compile(self, data):
        """Apply to the input sequence *data* each registered function

        and return the results as a dictionnary.



        :param data: Sequence of objects on which the statistics are computed.

        """

        values = tuple(self.key(elem) for elem in data)

        entry = dict()

        for key, func in self.functions.items():
            entry[key] = func(values)

        return entry


class MultiStatistics(dict):

    def compile(self, data):

        """Calls :meth:`Statistics.compile` with *data* of each

        :class:`Statistics` object.



        :param data: Sequence of objects on which the statistics are computed.

        """

        record = {}

        for name, stats in list(self.items()):
            record[name] = stats.compile(data)

        return record

    @property
    def fields(self):

        return sorted(self.keys())

    def register(self, name, function, *args, **kargs):

        for stats in list(self.values()):
            stats.register(name, function, *args, **kargs)


class Logbook(list):

    def __init__(self):

        super().__init__()
        self.buffindex = 0

        self.chapters = defaultdict(Logbook)

        self.columns_len = None

        self.header = None

        self.log_header = True

    def record(self, **infos):

        for key, value in list(infos.items()):

            if isinstance(value, dict):
                self.chapters[key].record(**value)

                del infos[key]

        self.append(infos)

    def select(self, *names):

        if len(names) == 1:
            return [entry.get(names[0], None) for entry in self]

        return tuple([entry.get(name, None) for entry in self] for name in names)

    @property
    def stream(self):

        startindex, self.buffindex = self.buffindex, len(self)

        return self.__str__(startindex)

    def __delitem__(self, key):

        if isinstance(key, slice):

            for i, in range(*key.indices(len(self))):

                self.pop(i)

                for chapter in list(self.chapters.values()):
                    chapter.pop(i)

        else:

            self.pop(key)

            for chapter in list(self.chapters.values()):
                chapter.pop(key)

    def pop(self, index=0):

        if index < self.buffindex:
            self.buffindex -= 1

        return super(self.__class__, self).pop(index)

    def __txt__(self, startindex):

        columns = self.header

        if not columns:
            columns = sorted(self[0].keys()) + sorted(self.chapters.keys())

        if not self.columns_len or len(self.columns_len) != len(columns):
            self.columns_len = list(map(len, columns))

        chapters_txt = {}

        offsets = defaultdict(int)

        for name, chapter in list(self.chapters.items()):

            chapters_txt[name] = chapter.__txt__(startindex)

            if startindex == 0:
                offsets[name] = len(chapters_txt[name]) - len(self)

        str_matrix = []

        for i, line in enumerate(self[startindex:]):

            str_line = []

            for j, name in enumerate(columns):

                if name in chapters_txt:

                    column = chapters_txt[name][i + offsets[name]]

                else:

                    value = line.get(name, "")

                    string = "{0:n}" if isinstance(value, float) else "{0}"

                    column = string.format(value)

                self.columns_len[j] = max(self.columns_len[j], len(column))

                str_line.append(column)

            str_matrix.append(str_line)

        if startindex == 0 and self.log_header:

            nlines = 1

            if len(self.chapters) > 0:
                nlines += max(list(map(len, list(chapters_txt.values())))) - len(self) + 1

            header = [[] for _ in range(nlines)]

            for j, name in enumerate(columns):

                if name in chapters_txt:

                    length = max(len(line.expandtabs()) for line in chapters_txt[name])

                    blanks = nlines - 2 - offsets[name]

                    for i in range(blanks):
                        header[i].append(" " * length)

                    header[blanks].append(name.center(length))

                    header[blanks + 1].append("-" * length)

                    for i in range(offsets[name]):
                        header[blanks + 2 + i].append(chapters_txt[name][i])

                else:

                    length = max(len(line[j].expandtabs()) for line in str_matrix)

                    for line in header[:-1]:
                        line.append(" " * length)

                    header[-1].append(name)

            str_matrix = chain(header, str_matrix)

        template = "\t".join("{%i:<%i}" % (i, l) for i, l in enumerate(self.columns_len))

        text = [template.format(*line) for line in str_matrix]

        return text

    def __str__(self, startindex=0):

        text = self.__txt__(startindex)

        return "\n".join(text)


class HallOfFame(object):

    def __init__(self, maxsize, similar=eq):

        self.maxsize = maxsize

        self.keys = list()

        self.items = list()

        self.similar = similar

    def update(self, population):

        """Update the hall of fame with the *population* by replacing the

        worst individuals in it by the best individuals present in

        *population* (if they are better). The size of the hall of fame is

        kept constant.



        :param population: A list of individual with a fitness attribute to

                           update the hall of fame with.

        """

        if len(self) == 0 and self.maxsize != 0:
            # Working on an empty hall of fame is problematic for the

            # "for else"

            self.insert(population[0])

        for ind in population:

            if ind.fitness > self[-1].fitness or len(self) < self.maxsize:

                for hofer in self:

                    # Loop through the hall of fame to check for any

                    # similar individual

                    if self.similar(ind, hofer):
                        break

                else:

                    # The individual is unique and strictly better than

                    # the worst

                    if len(self) >= self.maxsize:
                        self.remove(-1)

                    self.insert(ind)

    def insert(self, item):

        """Insert a new individual in the hall of fame using the

        :func:`~bisect.bisect_right` function. The inserted individual is

        inserted on the right side of an equal individual. Inserting a new

        individual in the hall of fame also preserve the hall of fame's order.

        This method **does not** check for the size of the hall of fame, in a

        way that inserting a new individual in a full hall of fame will not

        remove the worst individual to maintain a constant size.



        :param item: The individual with a fitness attribute to insert in the

                     hall of fame.

        """

        item = deepcopy(item)

        i = bisect_right(self.keys, item.fitness)

        self.items.insert(len(self) - i, item)

        self.keys.insert(i, item.fitness)

    def remove(self, index):

        """Remove the specified *index* from the hall of fame.



        :param index: An integer giving which item to remove.

        """

        del self.keys[len(self) - (index % len(self) + 1)]

        del self.items[index]

    def clear(self):

        """Clear the hall of fame."""

        del self.items[:]

        del self.keys[:]

    def __len__(self):

        return len(self.items)

    def __getitem__(self, i):

        return self.items[i]

    def __iter__(self):

        return iter(self.items)

    def __reversed__(self):

        return reversed(self.items)

    def __str__(self):

        return str(self.items)


class ParetoFront(HallOfFame):

    def __init__(self, similar=eq):

        HallOfFame.__init__(self, None, similar)

    def update(self, population):

        """Update the Pareto front hall of fame with the *population* by adding

        the individuals from the population that are not dominated by the hall

        of fame. If any individual in the hall of fame is dominated it is

        removed.



        :param population: A list of individual with a fitness attribute to

                           update the hall of fame with.

        """

        for ind in population:

            is_dominated = False

            dominates_one = False

            has_twin = False

            to_remove = []

            for i, hofer in enumerate(self):  # hofer = hall of famer

                if not dominates_one and hofer.fitness.dominates(ind.fitness):

                    is_dominated = True

                    break

                elif ind.fitness.dominates(hofer.fitness):

                    dominates_one = True

                    to_remove.append(i)

                elif ind.fitness == hofer.fitness and self.similar(ind, hofer):

                    has_twin = True

                    break

            for i in reversed(to_remove):  # Remove the dominated hofer

                self.remove(i)

            if not is_dominated and not has_twin:
                self.insert(ind)


"""
this part are change from deap, for form control
"""


class Terminal(object):
    __slots__ = ('name', 'value', 'conv_fct', 'dim', 'arity')

    def __init__(self, terminal, dim=1):
        self.dim = dim
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


class Primitive(object):
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


class PrimitiveTree(list):
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


class PrimitiveSet(object):

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

        prim = Primitive(name, arity)

        assert name not in self.context, "Primitives are required to have a unique name. " \
                                         "Consider using the argument 'name' to rename your " \
                                         "second '%s' primitive." % (name,)

        self.primitives.append(prim)
        self.context[prim.name] = primitive
        self.prims_count += 1

    def addTerminal(self, terminal, dim, name=None):

        if name is None and callable(terminal):
            name = str(terminal)

        assert name not in self.context, "Terminals are required to have a unique name. " \
                                         "Consider using the argument 'name' to rename your " \
                                         "second %s terminal." % (name,)

        if name is not None:
            self.context[name] = terminal
            self.dimtext[name] = dim

        prim = Terminal(terminal, dim)
        self.terminals.append(prim)
        self.terms_count += 1

    @property
    def terminalRatio(self):
        """Return the ratio of the number of terminals on the number of all
        kind of primitives.
        """
        return self.terms_count / float(self.terms_count + self.prims_count)


def pri(fu, *args, **kargs):
    return functools.partial(fu, *args, **kargs)


def sympy_prim_set(rep_name,
                   categories=("Add", "Mul", "Abs", "exp"),
                   name=None, dim=None, definate_operate=None, definate_variable=None, linkage=None,
                   partial_categories=None, index_categories=None, self_categories=None, ):
    """
        :type partial_categories: double list
        partial_categories = [["Add","Mul"],["x4"]]
        :type index_categories: list
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

    pset0 = PrimitiveSet('main')

    # define primitives
    if index_categories:
        for j, i in enumerate(index_categories):
            pset0.addPrimitive(functools.partial(sympy.Pow, e=i), arity=1, name='pow%s' % j)
    for i in categories:
        if i in functions2:
            pset0.addPrimitive(functions2[i], arity=2, name=i)
        if i in functions1:
            pset0.addPrimitive(functions1[i], arity=1, name=i)
    if partial_categories:

        for partial_categoriesi in partial_categories:
            for i in partial_categoriesi[0]:
                for j in partial_categoriesi[1]:
                    if i in ["Mul", "Add"]:
                        pset0.addPrimitive(pri(functions2[i], sympy.Symbol(j)), arity=1, name="{}_{}".format(i, j))
                    else:
                        pset0.addPrimitive(pri(functions2[i], right=sympy.Symbol(j)), arity=1,
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

    for sym, dimi in zip(rep_name, dim):
        pset0.addTerminal(sym, dim=dimi, name=str(sym))

    # define limit
    if linkage is None:
        linkage = [[]]
    assert isinstance(linkage, (list, tuple))
    if not isinstance(linkage[0], (list, tuple)):
        linkage = [linkage, ]

    dict_pri = dict(zip([_.name for _ in pset0.primitives], range(len(pset0.primitives))))
    dict_ter = dict(zip([_.name for _ in pset0.terminals], range(len(pset0.terminals))))

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
    for i, j in enumerate(pset0.primitives):
        print(i, j)
    for i, j in enumerate(pset0.terminals):
        print(i, j)

    return pset0


def produce(container1, generator1, random_state=None):
    return container1(generator1(random_state=random_state))


def generate(pset, min_=None, max_=None, random_state=None):
    if max_ is None:
        max_ = len(pset.terminals)
    if min_ is None:
        min_ = max_
    random_state = check_Random_state(random_state)

    pri2 = [i for i in pset.primitives if i.arity == 2]
    pri1 = [i for i in pset.primitives if i.arity == 1]

    max_varibale_set_long = max_
    varibale_set_long = random_state.randint(min_, max_varibale_set_long)
    '''random'''
    trem_set = random_state.sample(pset.terminals, varibale_set_long) * 20
    '''sequence'''
    trem_set = pset.terminals[:varibale_set_long] * 5
    init_operator_long = max_varibale_set_long * 3

    individual1 = []
    for i in range(init_operator_long):
        trem = random_state.choice(pri2) if random_state.random() > 0.5 * len(pri1) / len(
            pset.primitives) else random_state.choice(pri1)
        individual1.append(trem)
    individual2 = []
    for i in range(varibale_set_long):
        trem = trem_set[i]
        individual2.append(trem)
    # define protect primitives
    pri2 = [i for i in pset.primitives if i.arity == 2]
    protect_individual = []
    for i in range(varibale_set_long):
        trem = random_state.choice(pri2)
        protect_individual.append(trem)

    definate_operate = pset.definate_operate
    definate_variable = pset.definate_variable
    linkage = pset.linkage

    if definate_operate:
        for i in definate_operate:
            individual1[i[0]] = pset.primitives[random_state.choice(i[1])]

    if definate_variable:
        for i in definate_variable:
            individual2[i[0]] = pset.terminals[random_state.choice(i[1])]

    individual_all = protect_individual + individual1 + individual2
    if linkage:
        for i in linkage:
            for _ in i:
                individual_all[_] = individual_all[i[0]]

    return individual_all


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
        random_state = check_Random_state(3)
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


def initRepeat(container, func, n, random_state=None):
    max_int = np.iinfo(np.int32).max
    random_state = check_Random_state(random_state)
    seeds = random_state.sample(range(max_int), n)
    return container(func(random_state=check_Random_state(seeds[_])) for _ in range(n))


def my_custom_loss_func(y_true, y_pred):
    diff = - np.abs(y_true - y_pred) / y_true + 1
    return np.mean(diff)


mre_score = make_scorer(my_custom_loss_func, greater_is_better=True)


def score(individual, pset, x, y, score_method=explained_variance_score, add_coeff=True, filter_warning=True, **kargs):
    expr_no = sympy.sympify(compile_(individual, pset))
    # expr_no = sympy.expand(compile_(individual, pset), deep=False, power_base=False, power_exp=False, mul=True,
    #                        log=False, multinomial=False)
    if isinstance(expr_no, sympy.Mul) and len(expr_no.args) == 2:
        if isinstance(expr_no.args[0], sympy.Add) and expr_no.args[1].args == ():
            expr_no = sum([i * expr_no.args[1] for i in expr_no.args[0].args])
        if isinstance(expr_no.args[1], sympy.Add) and expr_no.args[0].args == ():
            expr_no = sum([i * expr_no.args[0] for i in expr_no.args[1].args])
        else:
            pass

    score, expr = calculate(expr_no, pset, x, y, score_method=score_method, add_coeff=add_coeff,
                            filter_warning=filter_warning, **kargs)
    return score, expr


def calculate(expr01, pset, x, y, score_method=explained_variance_score, add_coeff=True, filter_warning=True,
              del_no_important=False, **kargs):
    if filter_warning:
        warnings.filterwarnings("ignore")
    expr00 = deepcopy(expr01)
    if not score_method:
        score_method = r2_score
    if add_coeff:
        expr01, a_list = add_coefficient(expr01, **kargs)
        try:
            func0 = sympy.utilities.lambdify([_.value for _ in pset.terminals] + a_list, expr01)

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
                    cof_.append(np.round(cofi, decimals=2))
            cof = cof_
            for ai, choi in zip(a_list, cof):
                expr01 = expr01.subs(ai, choi)
        except (ValueError, NameError, TypeError):
            expr01 = expr00
    else:
        pass
    try:
        func0 = sympy.utilities.lambdify([_.value for _ in pset.terminals], expr01)
        re = func0(*x.T)
        assert_all_finite(re)
        check_array(re, ensure_2d=False)

    except (ValueError, DataConversionWarning, TypeError, NameError):
        score = -0
    else:
        def cv_expr(expr_):
            func0_ = sympy.utilities.lambdify([_.value for _ in pset.terminals], expr_)
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
                    func0 = sympy.utilities.lambdify([_.value for _ in pset.terminals], expri)
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


def cxoneooint_index(ind1, ind2, pset, random_state=None):
    linkage = pset.linkage
    random_state = check_Random_state(random_state)
    root = max(ind1.root, ind2.root)
    index = random_state.randint(root, root + len(ind1.pri_site))
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


def mutniform_index(ind1, pset, random_state=None):
    random_state = check_Random_state(random_state)
    ind10 = copy.copy(ind1)
    linkage = pset.linkage
    pri2 = [i for i in pset.primitives if i.arity == 2]
    pri1 = [i for i in pset.primitives if i.arity == 1]
    index = random_state.choice(ind10.pri_site)
    ind10[index] = random_state.choice(pri2) if random_state.random() > 0.5 * len(pri1) / len(
        pset.primitives) else random_state.choice(pri1)

    definate_operate = pset.definate_operate

    if definate_operate:
        for i in definate_operate:
            ind10[ind1.pri_site[i[0]]] = pset.primitives[random_state.choice(i[1])]

    if linkage:
        for i in linkage:
            for _ in i:
                ind10[_] = ind10[i[0]]
    return ind10,


def selRandom(individuals, k, random_state):
    random_state = check_Random_state(random_state)
    return [random_state.choice(individuals) for _ in range(k)]


def selTournament(individuals, k, tournsize, fit_attr="fitness", seed=None):
    seed = check_Random_state(seed)
    chosen = []
    for i in range(int(k // 5)):
        aspirants = selRandom(individuals, tournsize, seed)
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    chosen.extend(selRandom(individuals, 4 * int(k // 5), seed))
    return chosen


def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]
    random_state = check_Random_state(10)
    random_state.shuffle(offspring)
    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random_state.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random_state.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, pset=None, store=True):
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


@time_this_function
def main_part(x, y, pset, pop_n=100, random_seed=1, cxpb=0.8, mutpb=0.1, ngen=5, max_=None, **kargs):
    random_state = check_Random_state(random_seed)
    Fitness_ = create("Fitness_", Fitness, weights=(1.0,))
    PrimitiveTrees = create("PrimitiveTrees", PrimitiveTree, fitness=Fitness_)
    seed1, seed2, seed3, seed4, seed5, seed6, seed7 = [check_Random_state(i) for i in random_state.sample(range(10), 7)]

    toolbox = Toolbox()
    toolbox.register("generate_", generate, pset=pset, min_=None, max_=max_)
    toolbox.register("individual", produce, container1=PrimitiveTrees, generator1=toolbox.generate)
    toolbox.register('population', initRepeat, container=list, func=toolbox.individual, random_state=seed1)
    # def selection
    toolbox.register("selection", selTournament, tournsize=2, seed=seed2)

    # def mate
    toolbox.register("mate", cxoneooint_index, pset=pset, random_state=seed3)

    # def mutate
    toolbox.register("mutate", mutniform_index, pset=pset, random_state=seed4)

    # def elaluate
    toolbox.register("evaluate", score, pset=pset, x=x, y=y, score_method=r2_score, **kargs)

    stats = Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop = toolbox.population(n=pop_n)
    haln = 5
    hof = HallOfFame(haln)

    population, logbook = eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats,
                                   halloffame=hof, verbose=__debug__, pset=pset)

    return population, logbook, hof


# print(time.time())
# for i in range(10):
#     individual = generate_(pset1, random_state=check_Random_state(i), max_=60)
#     individual = PrimitiveTree(individual)
#     expr_no = sympy.sympify(compile_(individual, pset1))
#     cxOnePoint_index(individual, individual, pset1, random_state=None)
#     individual = mutUniForm_index(individual, pset1, random_state=None)[0]
#     # expr, score1 = calculate(expr_no, pset1, x, y, score_method=explained_variance_score, add_coeff=True)
#     expr, score1 = score(individual, pset1, x, y, score_method=explained_variance_score, add_coeff=True,
#     filter_warning=True)
# print(time.time())

# -*- coding: utf-8 -*-


if __name__ == "__main__":
    from feature_toolbox.tools.data_import import Dataset

    store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp_last')
    data = Dataset(r'C:\Users\Administrator\Desktop\band_gap_exp_last')
    all_import_structure = data.csv.all_import_structure
    data_import = all_import_structure.drop(
        ["name", "structure", "structure_type", "space_group", "reference", 'material_id', 'composition'], axis=1)

    data216_import = data_import.iloc[np.where(data_import['group_number'] == 216)[0]]
    data225_import = data_import.iloc[np.where(data_import['group_number'] == 225)[0]]
    data186_import = data_import.iloc[np.where(data_import['group_number'] == 186)[0]].drop("BeO186", axis=0)
    data216_225_import = pd.concat((data216_import, data225_import))
    data = data225_import
    y = data["exp_gap"].values
    x_data = data.drop(["exp_gap", "group_number"], axis=1)
    x = x_data.values
    # scal = preprocessing.MinMaxScaler()
    # x = scal.fit_transform(x)
    # x, y = utils.shuffle(x, y, random_state=5)
    name, rep_name = get_name(x_data)
    # pset1 = sympy_prim_set(
    #     categories=('Add', 'Sub', 'Mul', 'Div', 'Max', "Rec", 'exp', "log", "Abs"),
    #     name=name,
    #     partial_categories=None,
    #     rep_name=rep_name,
    #     index_categories=(1 / 3, 1 / 2, 1, 2, 3),
    #     definate_operate=[
    #         [-11, ['log']],
    #         [-10, ['Mul']],
    #         [-9, ['Mul']],
    #         [-8, [2]],
    #         [-7, ['Add', 'Sub', 'Mul', 'Div']],
    #         [-6, ['Div']],
    #         [-5, ["Rec"]],
    #         [-4, [0, 1, 2, 3, 4, "Rec", 'exp', "log", "Abs"]],
    #         [-3, [0, 1, 2, 3, 4, "Rec", 'exp', "log", "Abs"]],
    #         # [-4, ["Rec"]],
    #         # [-3, ["Rec"]],
    #         [-1, [2]],
    #         [-2, [2]],
    #     ],
    #     definate_variable=[
    #
    #         [-5, [0]],
    #         [-4, [6]],
    #         [-3, [7]],
    #         [-1, [24]],
    #         [-2, [25]],
    #     ],
    #     linkage=[[-6, -7], [-8, -9]])
    pset1 = sympy_prim_set(
        categories=('Add', 'Sub', 'Mul', 'Div', 'Max', "Rec", 'exp', "log", "Abs"),
        name=name,
        partial_categories=None,
        rep_name=rep_name,
        index_categories=(1 / 3, 1 / 2, 1, 2, 3),
        definate_operate=[
            [-9, ['Mul', 'Div']],
            [-8, ['Mul', 'Div']],
            [-7, ['Add', 'Sub', 'Mul', 'Div']],
            [-6, ['Add', 'Sub', 'Mul', 'Div']],
            [-5, ["Rec"]],
            [-4, [0, 1, 2, 3, 4, "Rec", 'exp', "log", "Abs"]],
            [-3, [0, 1, 2, 3, 4, "Rec", 'exp', "log", "Abs"]],
            # [-4, ["Rec"]],
            # [-3, ["Rec"]],
            [-1, [2]],
            [-2, [2]],
        ],
        definate_variable=[

            [-5, [0]],
            [-4, [6]],
            [-3, [7]],
            [-1, [18]],
            [-2, [19]],
        ],
        linkage=[[-6, -7], [-8, -9]])

    result = main_part(x, y, pset1, pop_n=500, random_seed=1, cxpb=0.8, mutpb=0.1, ngen=10, max_=5,
                       inter_add=True, iner_add=False, random_add=False)
