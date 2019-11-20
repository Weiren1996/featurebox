# -*- coding: utf-8 -*-

# @Time    : 2019/11/12 14:21
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
"""
All part are copy from deap (https://github.com/DEAP/deap)
"""
import copy
import functools
import random
import warnings

import sympy

warnings.filterwarnings("ignore")


class FixedTerminal(object):
    """

    """
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
    """

    """
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


class FixedSet(object):
    """

    """

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

        assert name not in self.context, "Primitives are required to have a unique x_name. " \
                                         "Consider using the argument 'x_name' to rename your " \
                                         "second '%s' primitive." % (name,)

        self.primitives.append(prim)
        self.context[prim.name] = primitive
        self.prims_count += 1

    def addTerminal(self, terminal, name=None, dim=None):

        if name is None and callable(terminal):
            name = str(terminal)

        assert name not in self.context, "Terminals are required to have a unique x_name. " \
                                         "Consider using the argument 'x_name' to rename your " \
                                         "second %s terminal." % (name,)

        if name is not None:
            self.context[name] = terminal

        prim = FixedTerminal(terminal)
        self.terminals.append(prim)
        self.terms_count += 1
        self.dimtext[name] = dim

    @property
    def terminalRatio(self):
        """Return the ratio of the number of terminals on the number of all
        kind of primitives.
        """
        return self.terms_count / float(self.terms_count + self.prims_count)


class FixedTree(list):
    """

    """
    hasher = str

    def __init__(self, content):
        list.__init__(self, content)

        assert sum(_.arity - 1 for _ in self.primitives) + 1 >= len(self.terminals)
        assert len(self.terminals) >= 2

    @property
    def root(self):
        """

        Returns
        -------
        start site number of tree
        """
        len_ter = len(self.terminals) - 1
        num_pri = list(range(len(self.primitives)))
        num_pri.reverse()
        i = 0
        for i in num_pri:
            if len_ter == 0:
                break
            elif len_ter <= 0:
                raise ("Add terminals or move back the {}".format(self[i - 1]),
                       "because the {} have insufficient terminals, "
                       "need {},but get {}".format(self[i - 1], self[i - 1].arity, len_ter - self[i - 1].arity)
                       )
            len_ter = len_ter - self[i].arity + 1
        if self[i].arity == 1:
            return i
        else:
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
        return self.str_run_index(self.root)

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

    @classmethod
    def search_y_name(cls, name):
        """

        Parameters
        ----------
        name

        Returns
        -------

        """
        list_name = []
        for i in range(len(cls)):
            if cls[i].name == name:
                list_name.append(i)
        return list_name

    @property
    def number_branch(self):
        """

        Returns
        -------
        dict,a site tree
        """

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

    def number_branch_index(self, index):
        """
        Returns
        -------
        number_branch for given index
        """
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
        """
        Returns
        -------
        str of tree by given number_branch,default is the str of root number_branch
        """
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

    def str_run_index(self, i):
        return self.str_run(self.number_branch_index(i))

    def indexs_in_node(self, i):
        """

        Returns
        -------
        indexs in node branch
        """

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


def FixedSetFill(x_name, rep_name=None, power_categories=None, categories=("Add", "Mul", "Abs", "exp"),
                 partial_categories=None, self_categories=None, dim=None, max_=5,
                 definate_operate=None, definate_variable=None, operate_linkage=None, variable_linkage=None):
    """

    Parameters
    ----------
    :type partial_categories: double list
    partial_categories = [["Add","Mul"],["x4"]]
    :type power_categories: list
    index_categories=[0.5,1,2,3]
    :type dim: list,tuple
    :type x_name: list,tuple
    :type max_
    :type rep_name: list,tuple
    :type categories: list,tuple
    :param self_categories:
    def rem(a):
        return 1-a
    self_categories = [[rem, 1, 'rem']]
    :type definate_variable: list,tuple
    definate_variable = [(-1, [1, ]), ]
    :type definate_operate: list,tuple
    definate_operate = [(-1, [0, ]), ]
    :param operate_linkage
    :param variable_linkage
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

    functions2 = {"Add": sympy.Add, 'Sub': Sub, 'Mul': sympy.Mul, 'Div': Div}
    functions1 = {"sin": sympy.sin, 'cos': sympy.cos, 'exp': sympy.exp, 'log': sympy.ln,
                  'Abs': sympy.Abs, "Neg": functools.partial(sympy.Mul, -1.0),
                  "Rec": functools.partial(sympy.Pow, e=-1.0),
                  'Zeroo': zeroo, "Oneo": oneo, "Rem": rem, "Self": self}

    pset0 = FixedSet('main')

    if power_categories:
        for j, i in enumerate(power_categories):
            pset0.addPrimitive(functools.partial(sympy.Pow, e=i), arity=1, name='pow%s' % j)

    for i in categories:
        if i in functions1:
            pset0.addPrimitive(functions1[i], arity=1, name=i)
        if i in functions2:
            pset0.addPrimitive(functions2[i], arity=2, name=i)

    if partial_categories:
        for partial_categoriesi in partial_categories:
            for i in partial_categoriesi[0]:
                for j in partial_categoriesi[1]:
                    if i in ["Mul", "Add"]:
                        pset0.addPrimitive(functools.partial(functions2[i], sympy.Symbol(j)), arity=1,
                                           name="{}_{}".format(i, j))
                    elif i in ["Div", "Sub"]:
                        pset0.addPrimitive(functools.partial(functions2[i], right=sympy.Symbol(j)), arity=1,
                                           name="{}_{}".format(i, j))
                    else:
                        pass
    if self_categories:
        for i in self_categories:
            pset0.addPrimitive(i[0], i[1], i[2])

    # define terminal
    if not rep_name:
        rep_name = ['{}{}'.format("x", i) for i, _ in enumerate(x_name)]

    rep_name = [sympy.Symbol(i) for i in rep_name]

    if dim is None:
        dim = [None] * len(rep_name)

    assert len(dim) == len(x_name) == len(rep_name)

    for sym, dimi in zip(rep_name, dim):
        pset0.addTerminal(sym, name=str(sym), dim=dimi)

    dict_pri = dict(zip([_.name for _ in pset0.primitives], range(len(pset0.primitives))))
    dict_ter = dict(zip([_.name for _ in pset0.terminals], range(len(pset0.terminals))))

    if max_ is None:
        max_ = len(pset0.terminals)

    # define limit
    def link_check(checking_linkage):
        """"""
        if checking_linkage is None:
            checking_linkage = [[]]
        assert isinstance(checking_linkage, (list, tuple))
        if not isinstance(checking_linkage[0], (list, tuple)):
            checking_linkage = [checking_linkage, ]
        return checking_linkage

    operate_linkage = link_check(operate_linkage)
    variable_linkage = link_check(variable_linkage)

    operate_linkage = [[j - max_ for j in i] for i in operate_linkage]
    linkage = operate_linkage + variable_linkage

    if definate_operate:
        definate_operate = [list(i) for i in definate_operate]
        for i, j in enumerate(definate_operate):
            j = list(j)
            definate_operate[i][1] = [dict_pri[_] if _ in dict_pri else _ for _ in j[1]]
        err_str = [i for j in definate_operate for i in j[1] if isinstance(i, str)]
        if len(err_str) > 0:
            raise NameError("{} in definate_operate sholud be defined in categories".format(set(err_str)))
    if definate_variable:
        definate_variable = [list(i) for i in definate_variable]
        for i, j in enumerate(definate_variable):
            j = list(j)
            definate_variable[i][1] = [dict_ter[_] if _ in dict_ter else _ for _ in j[1]]

    pset0.definate_operate = definate_operate
    pset0.definate_variable = definate_variable
    pset0.linkage = linkage

    pset0.rep_name_list = rep_name
    pset0.real_name_list = x_name
    pset0.dim_list = dim

    pset0.max_ = max_
    print(dict_pri)
    print(dict_ter)

    return pset0


def generate_index(pset, min_=None, max_=None):
    """

    Parameters
    ----------
    pset
    min_
    max_

    Returns
    -------

    """
    if max_ is None:
        max_ = len(pset.terminals)
    _ = min_
    min_ = max_

    pri2 = [i for i in pset.primitives if i.arity == 2]
    pri1 = [i for i in pset.primitives if i.arity == 1]

    max_varibale_set_long = max_
    varibale_set_long = random.randint(min_, max_varibale_set_long)
    '''random'''
    trem_set = random.sample(pset.terminals, varibale_set_long) * 20
    '''sequence'''
    # trem_set = pset.terminals[:varibale_set_long] * 20

    init_operator_long = max_varibale_set_long * 4
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


def cxOnePoint_index(ind1, ind2, pset):
    """

    Parameters
    ----------
    ind1
    ind2
    pset

    Returns
    -------

    """
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


def mutUniForm_index(ind10, pset, ):
    """

    Parameters
    ----------
    ind1
    pset

    Returns
    -------

    """
    linkage = pset.linkage
    definate_operate = pset.definate_operate
    ranges = list(range(ind10.pri_site[-1] + 1))
    pri2 = [i for i in pset.primitives if i.arity == 2]
    pri1 = [i for i in pset.primitives if i.arity == 1]
    index = random.choice(ind10.pri_site)
    ind10[index] = random.choice(pri2) if random.random() > 0.5 * len(pri1) / len(
        pset.primitives) else random.choice(pri1)

    if definate_operate:
        for i in definate_operate:
            ind10[ranges[i[0]]] = pset.primitives[random.choice(i[1])]

    if linkage:
        for i in linkage:
            for _ in i:
                ind10[_] = ind10[i[0]]

    return ind10,
