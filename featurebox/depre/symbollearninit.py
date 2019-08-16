#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/28 16:26
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

import functools
import operator
import random
import sys
import warnings
from copy import deepcopy
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
import sympy
from deap import creator, base, tools, gp
from deap.algorithms import varAnd
from deap.gp import PrimitiveSet, genHalfAndHalf
from deap.tools import Logbook, selTournament, Statistics, HallOfFame, MultiStatistics
from scipy import optimize
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import r2_score, explained_variance_score, make_scorer
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import assert_all_finite, check_array

from featurebox.tools.exports import Store
from featurebox.tools.imports import Call
from featurebox.tools.tool import parallize, time_this_function, check_random_state

warnings.filterwarnings("ignore")
"""
this is a description
"""


def pri(fu, *args, **kargs):
    return functools.partial(fu, *args, **kargs)


def sympy_prim_set(rep_name,
                   categories=("Add", "Mul", "Abs", "exp"),
                   name=None, dim=None, partial_categories=None, index_categories=None, self_categories=None, ):
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

    pset0 = PrimitiveSet('main', 0)

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

    for sym in rep_name:
        pset0.addTerminal(sym, name=str(sym))

    pset0.rep_name_list = rep_name
    pset0.name_list = name
    pset0.dim_list = dim
    for i, j in enumerate(pset0.primitives[object]):
        print(i, j.name)
    for i, j in enumerate(pset0.terminals[object]):
        print(i, j.name)

    return pset0


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
    # if isinstance(expr01, sympy.Add):
    #     for i, j in enumerate(expr01.args):
    #         Wi = sympy.Symbol("W%s" % i)
    #         expr01 = expr01.subs(j, Wi * j)
    #         a_list.append(Wi)
    # else:
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
    diff = np.mean(diff)
    if diff > 1:
        diff = 0
    return diff


mre_score = make_scorer(my_custom_loss_func, greater_is_better=True)


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


def sub(expr01, subed, subs):
    listt = list(zip(subed, subs))
    return expr01.subs(listt)


def score(individual, pset, x, y, score_method=explained_variance_score, add_coeff=True, filter_warning=True, **kargs):
    expr_no = sympy.sympify(compile_(individual, pset))

    # expr_no = sympy.expand(compile_(individual, pset), deep=False, power_base=False, power_exp=False, mul=True,
    #                        log=False, multinomial=False)
    score, expr = calculate(expr_no, pset, x, y, score_method=score_method, add_coeff=add_coeff,
                            filter_warning=filter_warning, **kargs)

    return score, expr


def calculate(expr01, pset, x, y, score_method=explained_variance_score, add_coeff=True, filter_warning=True,
              del_no_important=False, **kargs):
    terminals = pset.terminals[object]
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
                    cof_.append(np.round(cofi, decimals=2))
            cof = cof_
            for ai, choi in zip(a_list, cof):
                expr01 = expr01.subs(ai, choi)
        except (ValueError, NameError, TypeError, KeyError):
            expr01 = deepcopy(expr00)
    else:
        pass
    try:
        func0 = sympy.utilities.lambdify([_.value for _ in terminals], expr01)
        re = func0(*x.T)
        assert_all_finite(re)
        check_array(re, ensure_2d=False)

    except (ValueError, DataConversionWarning, TypeError, NameError, KeyError):
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


def multieaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
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

        return (a * 82 + 99 * b) * (1 - alpha * abs(a - b)) / 181

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

        def funcc(a, b):
            return (a * 82 + 99 * b) * (1 - alpha * abs(a - b)) / 181

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


@time_this_function
def main_part(x, y, x2, y2, pset, pop_n=100, cxpb=0.8, mutpb=0.1, ngen=5, max_=None, random_seed=0, mut_max=3, alpha=1,
              tournsize=3, max_value=10, **kargs):
    random.seed(random_seed)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", ExpressionTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("generate", genHalfAndHalf, pset=pset, min_=2, max_=max_)
    toolbox.register("individual", tools.initIterate, container=creator.Individual, generator=toolbox.generate)
    toolbox.register('population', tools.initRepeat, container=list, func=toolbox.individual)
    # def selection
    toolbox.register("selection", selTournament, tournsize=tournsize)
    # def mate
    toolbox.register("mate", gp.cxOnePoint)

    # def mutate
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=mut_max)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # def elaluate
    toolbox.register("evaluate", score, pset=pset, x=x, y=y, score_method=my_custom_loss_func, **kargs)
    toolbox.register("evaluate2", score, pset=pset, x=x2, y=y2, score_method=my_custom_loss_func, **kargs)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_value))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_value))
    stats = MultiStatistics(score=Statistics(lambda ind: ind.fitness.values[0]),
                            )
    # stats.register("avg", np.mean)
    # stats.register("std", np.std)
    # stats.register("min", np.min)
    stats.register("max", np.max)

    pop = toolbox.population(n=pop_n)
    haln = 2
    hof = HallOfFame(haln)

    population, logbook = multieaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, alpha=alpha,
                                   halloffame=hof, verbose=__debug__, pset=pset)
    return population, logbook, hof


if __name__ == "__main__":

    store = Store(r'C:\Users\Administrator\Desktop\wencheng')
    data = Call(r'C:\Users\Administrator\Desktop\wencheng')
    data_import = data.csv.wencheng

    BCC = data_import.iloc[np.where(data_import.index == "BCC")[0]]
    FCC = data_import.iloc[np.where(data_import.index == "FCC")[0]]

    x = FCC.values[:, :-1]
    y = FCC.values[:, -1]
    x2 = BCC.values[:, :-1]
    y2 = BCC.values[:, -1]
    # scal = preprocessing.MinMaxScaler()
    # x = scal.fit_transform(x)
    # x, y = utils.shuffle(x, y, random_state=5)
    name, rep_name = get_name(x)
    pset1 = sympy_prim_set(
        categories=('Add', 'Sub', 'Mul', 'Div', "Rec", 'exp'),
        name=name,
        partial_categories=None,
        rep_name=rep_name,
        index_categories=(1 / 3, 1 / 2, 1, 2, 3), )

    # result = mainPart(x, y, x2, y2, pset1, pop_n=500, random_seed=1, cxpb=0.8, mutpb=0.1, ngen=50, max_=5,
    #                    mut_max=3, tournsize=3, max_value=5,
    #                    inter_add=False, iner_add=False, random_add=False)
    alpha = [0, 0.5, 1, 1.5]
    mut_max = [2, 3]
    tournsize = [2, 3]
    mutpb = [0.1, 0.2]
    max_value = [5, 6, 7]
    cxpb = [0.7, 0.8, 0.9]

    for n, arg in enumerate(product(alpha, mut_max, tournsize, mutpb, max_value, cxpb)):
        i, al, j, k, l, m = arg

        result = main_part(x, y, x2, y2, pset1, pop_n=50, random_seed=1,
                           cxpb=m, ngen=50, max_=5, mutpb=k,
                           mut_max=i, tournsize=j, max_value=l, alpha=al,
                           inter_add=False, iner_add=False, random_add=False)

        file = open('test.txt', 'a')
        file.write("\n\nTest{},Args:mut_max={},tournsize={},mutpb={},max_value={},cxpb={},alpha={}\n".format(n, *arg))
        file.write(str(result[1]))
        file.close()
