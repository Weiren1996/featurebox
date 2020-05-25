# -*- coding: utf-8 -*-

# @Time    : 2019/11/11 13:15
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import functools

import sympy
from deap.gp import PrimitiveTree, PrimitiveSet


class ExpressionTree(PrimitiveTree):
    """

    """
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


class ExpressionSet(PrimitiveSet):

    def __init__(self, set_name, prefix="x"):
        super(ExpressionSet, self).__init__(set_name, 0, prefix=prefix)
        self.dimtext = {"__builtins__": None}

    def addTerminal(self, terminal, name=None, dim=None):
        PrimitiveSet.addTerminal(self, terminal, name=name)
        if name is None:
            name = terminal.__name__
        self.dimtext[name] = dim


def ExpressionSetFill(x_name, rep_name=None, power_categories=None, categories=("Add", "Mul", "Self", "exp"),
                      partial_categories=None, self_categories=None, dim=None):
    """

    Parameters
    ----------
        :type partial_categories: double list
        partial_categories = [["Add","Mul"],["x4"]]
        :type power_categories: list
        index_categories=[0.5,1,2,3]
        :type dim: list,tuple
        :type x_name: list,tuple
        :type rep_name: list,tuple
        :type categories: list,tuple
        :param self_categories:
        def rem(a):
            return 1-a
        self_categories = [[rem, 1, 'rem']]
    """

    def Div(left, right):
        return left / right

    def Sub(left, right):
        return left - right

    def zeroo(_):
        return 0

    def oneo(_):
        return 1

    def remo(a):
        return 1 - a

    def self(_):
        return _

    functions2 = {"Add": sympy.Add, 'Sub': Sub, 'Mul': sympy.Mul, 'Div': Div}
    functions1 = {"sin": sympy.sin, 'cos': sympy.cos, 'exp': sympy.exp, 'log': sympy.ln,
                  'Abs': sympy.Abs, "Neg": functools.partial(sympy.Mul, -1.0),
                  "Rec": functools.partial(sympy.Pow, e=-1.0),
                  'Zeroo': zeroo, "Oneo": oneo, "Remo": remo, "Self": self}

    pset0 = ExpressionSet('main')

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

    pset0.rep_name_list = rep_name
    pset0.real_name_list = x_name
    pset0.dim_list = dim

    return pset0
