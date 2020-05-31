# -*- coding: utf-8 -*-

# @Time    : 2019/11/12 15:37
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import functools
import operator
import random
import sys
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import sympy
from deap.tools import Logbook

from scipy import optimize
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import r2_score
from sklearn.utils import check_array
from sympy import sympify

from featurebox.combination.dim import dim_func, dnan, Dim
from featurebox.tools.exports import Store


def calcualte_dim(expr01, rep_list, dim_list):
    func0 = sympy.utilities.lambdify(rep_list, expr01, modules=[dim_func(), 'numpy'])
    dim_ = func0(*dim_list)
    return dim_


def _compile(expr_, pset):
    """

    Parameters
    ----------
    expr_
    pset

    Returns
    -------

    """
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


def addCoefficient(expr01, inter_add=True, iner_add=True, random_add=None):
    """

    Parameters
    ----------
    expr01
    inter_add
    iner_add
    random_add

    Returns
    -------

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
    a_list = []
    #

    if isinstance(expr01, sympy.Add):

        for i, j in enumerate(expr01.args):
            Wi = sympy.Symbol("W%s" % i)
            expr01 = expr01.subs(j, Wi * j)
            a_list.append(Wi)

    else:

        A = sympy.Symbol("A")
        expr01 = sympy.Mul(expr01, A)

        a_list.append(A)

    if inter_add:
        B = sympy.Symbol("B")
        expr01 = expr01 + B
        a_list.append(B)

    if iner_add:
        cho_add = [i.args for i in arg_list if isinstance(i, sympy.Add)]
        cho_add = [[_ for _ in cho_addi if not _.is_number] for cho_addi in cho_add]
        [cho.extend(i) for i in cho_add]

    if random_add:
        pass
    #     lest = [i for i in arg_list if i not in cho]
    #     if len(lest) != 0:
    #         cho2 = random.sample(lest, 1)
    #         cho.extend(cho2)

    a_cho = [sympy.Symbol("k%s" % i) for i in range(len(cho))]
    for ai, choi in zip(a_cho, cho):
        expr01 = expr01.subs(choi, ai * choi)
    a_list.extend(a_cho)

    return expr01, a_list


def calculateExpr(expr01, x, y, terminals, scoring=None, add_coeff=True,
                  del_no_important=False, filter_warning=True, inter_add=True, iner_add=True, random_add=None):
    """

    Parameters
    ----------
    random_add
    iner_add
    inter_add
    expr01
    x
    y
    scoring
    add_coeff
    del_no_important
    filter_warning
    terminals

    Returns
    -------

    """

    def split_x(x):
        if x.ndim == 1:
            return [x]
        else:
            return [*x.T]

    if filter_warning:
        warnings.filterwarnings("ignore")
    if not scoring:
        scoring = r2_score

    expr00 = deepcopy(expr01)  #

    if add_coeff:

        expr01, a_list = addCoefficient(expr01, inter_add=inter_add, iner_add=iner_add, random_add=random_add)

        try:
            func0 = sympy.utilities.lambdify(terminals + a_list, expr01)

            def func(x_, p):
                """"""
                num_list = []

                num_list.extend(split_x(x))

                num_list.extend(p)
                return func0(*num_list)

            def res(p, x_, y_):
                """"""
                return y_ - func(x_, p)

            result = optimize.least_squares(res, x0=[1] * len(a_list), args=(x, y), loss='linear', ftol=1e-3)

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
        except (ValueError, KeyError, NameError, TypeError):
            expr01 = expr00

    else:
        pass  #

    try:
        if del_no_important and isinstance(expr01, sympy.Add) and len(expr01.args) >= 3:
            re_list = []
            for expri in expr01.args:
                if not isinstance(expri, sympy.Float):
                    func0 = sympy.utilities.lambdify(terminals, expri)
                    re = np.mean(func0(*split_x(x)))
                    if abs(re) > abs(0.001 * np.mean(y)):
                        re_list.append(expri)
                else:
                    re_list.append(expri)
            expr01 = sum(re_list)
        else:
            pass

        func0 = sympy.utilities.lambdify(terminals, expr01)
        try :
            re = func0(*split_x(x))
        except:
            print(expr01)
        sp = split_x(x)
        re = func0(*sp)
        re = re.ravel()
        assert y.shape == re.shape
        # assert_all_finite(re)
        re = check_array(re, ensure_2d=False)
        score = scoring(y, re)

    except (ValueError, DataConversionWarning, NameError, KeyError, AssertionError, AttributeError):
        score = -0
    else:
        if np.isnan(score):
            score = -0
    return score, expr01


def calculatePrecision(individual, pset, x, y, scoring=None, add_coeff=True, filter_warning=True, cal_dim=True,
                       inter_add=True, iner_add=True, random_add=None):
    """

    Parameters
    ----------
    inter_add
    iner_add
    random_add
    cal_dim
    scoring
    individual
    pset
    x
    y
    add_coeff
    filter_warning

    Returns
    -------

    """

    if scoring is None:
        scoring = r2_score
    # '''1 not expand'''
    expr_no = sympify(_compile(individual, pset))
    # '''2 expand by sympy.expand,long expr is slow, use if when needed'''
    # expr_no = sympy.expand(compile_(individual, pset), deep=False, power_base=False, power_exp=False, mul=True,
    #                        log=False, multinomial=False)

    terminals = pset.rep_name_list

    score, expr = calculateExpr(expr_no, x, y, terminals=terminals, scoring=scoring, add_coeff=add_coeff,
                                filter_warning=filter_warning, inter_add=inter_add, iner_add=iner_add,
                                random_add=random_add)

    if cal_dim:
        rep_list = pset.rep_name_list
        dim_list = pset.dim_list
        try:
            dim = calcualte_dim(expr, rep_list, dim_list)
        except TypeError:
            dim = dnan
    else:
        dim = dnan

    if isinstance(dim, (float, int)) or dim is None:
        dim = dnan
        withdim = 0

    elif isinstance(dim, Dim) and not dim.anyisnan:
        withdim = 1
    else:
        dim = dnan
        withdim = 0

    return score, expr, dim, withdim


def custom_loss_func(y_true, y_pred):
    """"""
    diff = - np.abs(y_true - y_pred) / y_true + 1
    return np.mean(diff)


def getName(x):
    """"""
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


def sub(expr01, subed, subs):
    """"""
    listt = list(zip(subed, subs))
    return expr01.subs(listt)


def varAnd(population, toolbox, cxpb, mutpb):
    rst = random.getstate()
    offspring = [toolbox.clone(ind) for ind in population]
    random.setstate(rst)
    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
    return offspring


# def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
#              halloffame=None, verbose=__debug__, pset=None, store=True):
#     """
#
#     Parameters
#     ----------
#     population
#     toolbox
#     cxpb
#     mutpb
#     ngen
#     stats
#     halloffame
#     verbose
#     pset
#     store
#     Returns
#     -------
#
#     """
#     rst = random.getstate()
#     len_pop = len(population)
#     logbook = Logbook()
#     logbook.header = ['gen', 'pop'] + (stats.fields if stats else [])
#
#     # Evaluate the individuals with an invalid fitness
#     invalid_ind = [ind for ind in population if not ind.fitness.valid]
#
#     # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
#     fitnesses = toolbox.parallel(iterable=population)
#     for ind, fit, in zip(invalid_ind, fitnesses):
#         ind.fitness.values = fit[0],
#         ind.expr = fit[1]
#         ind.dim = fit[2]
#         ind.withdim = fit[3]
#
#     add_ind = toolbox.select_kbest_target_dim(population, K_best=0.1 * len_pop)
#     if halloffame is not None:
#         halloffame.update(add_ind)
#
#     record = stats.compile(population) if stats else {}
#     logbook.record(gen=0, nevals=len(population), **record)
#     if verbose:
#         print(logbook.stream)
#     data_all = {}
#
#     # Begin the generational process
#     random.setstate(rst)
#     for gen in range(1, ngen + 1):
#         rst = random.getstate()
#
#         if store:
#             rst = random.getstate()
#             target_dim = toolbox.select_kbest_target_dim.keywords['dim_type']
#             subp = functools.partial(sub, subed=pset.rep_name_list, subs=pset.real_name_list)
#             datamnist = {"gen{}_pop{}".format(gen, n): {"gen": gen, "pop": n,
#                                                    "score": i.fitness.values[0],
#                                                    "expr": str(subp(i.expr)),
#                                                    "with_dim": 1 if i.withdim else 0,
#                                                    "dim_is_target_dim": 1 if i.dim in target_dim else 0,
#                                                    "gen_dim": "{}{}".format(gen, 1 if i.withdim else 0),
#                                                    "gen_target_dim": "{}{}".format(gen,
#                                                                                    1 if i.dim in target_dim else 0),
#                                                    "socre_dim": i.fitness.values[0] if i.withdim else 0,
#                                                    "socre_target_dim": i.fitness.values[
#                                                        0] if i.dim in target_dim else 0,
#                                                    } for n, i in enumerate(population) if i is not None}
#             data_all.update(datamnist)
#         random.setstate(rst)
#         # select_gs the next generation individuals
#         offspring = toolbox.select_gs(population, len_pop)
#
#         # Vary the pool of individuals
#         offspring = varAnd(offspring, toolbox, cxpb, mutpb)
#
#         rst = random.getstate()
#
#         # Evaluate the individuals with an invalid fitness
#         invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#         # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
#         # fitnesses = parallelize(n_jobs=3, func=toolbox.evaluate, iterable=invalid_ind,  respective=False)
#         fitnesses = toolbox.parallel(iterable=invalid_ind)
#         for ind, fit in zip(invalid_ind, fitnesses):
#             ind.fitness.values = fit[0],
#             ind.expr = fit[1]
#             ind.dim = fit[2]
#             ind.withdim = fit[3]
#
#         add_ind = toolbox.select_kbest_target_dim(population, K_best=0.1 * len_pop)
#         # add_ind2 = toolbox.select_kbest_dimless(population, K_best=0.2 * len_pop)
#         # add_ind3 = toolbox.select_kbest(population, K_best=5)
#         offspring += add_ind
#         # offspring += add_ind2
#         # offspring += add_ind3
#
#         # Update the hall of fame with the generated individuals
#         if halloffame is not None:
#             halloffame.update(add_ind)
#             if len(halloffame.items) > 0 and halloffame.items[-1].fitness.values[0] >= 0.95:
#                 print(halloffame.items[-1])
#                 print(halloffame.items[-1].fitness.values[0])
#                 break
#
#         population[:] = offspring
#         # Append the current generation statistics to the logbook
#         record = stats.compile(population) if stats else {}
#         logbook.record(gen=gen, nevals=len(population), **record)
#         if verbose:
#             print(logbook.stream)
#         random.setstate(rst)
#
#     store = Store()
#     store.to_csv(data_all)
#     return population, logbook

def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, pset=None, store=True):
    """

    Parameters
    ----------
    population
    toolbox
    cxpb
    mutpb
    ngen
    stats
    halloffame
    verbose
    pset
    store
    Returns
    -------

    """
    rst = random.getstate()
    len_pop = len(population)
    logbook = Logbook()
    logbook.header = [] + (stats.fields if stats else [])
    data_all = {}
    random.setstate(rst)

    for gen in range(1, ngen + 1):
        "评价"
        rst = random.getstate()
        """score"""
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.parallel(iterable=population)
        for ind, fit, in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0],
            ind.expr = fit[1]
            ind.y_dim = fit[2]
            ind.withdim = fit[3]
        random.setstate(rst)

        rst = random.getstate()
        """elite"""
        add_ind = []
        add_ind1 = toolbox.select_kbest_target_dim(population, K_best=0.1 * len_pop)
        add_ind2 = toolbox.select_kbest_dimless(population, K_best=0.1 * len_pop)
        add_ind3 = toolbox.select_kbest(population, K_best=5)
        add_ind += add_ind1
        add_ind += add_ind2
        add_ind += add_ind3
        elite_size = len(add_ind)
        random.setstate(rst)

        rst = random.getstate()
        """score"""
        if store:
            target_dim = toolbox.select_kbest_target_dim.keywords['dim_type']
            subp = functools.partial(sub, subed=pset.rep_name_list, subs=pset.real_name_list)
            data = {"gen{}_pop{}".format(gen, n): {"gen": gen, "pop": n,
                                                   "score": i.fitness.values[0],
                                                   "expr": str(subp(i.expr)),
                                                   "with_dim": 1 if i.withdim else 0,
                                                   "dim_is_target_dim": 1 if i.y_dim in target_dim else 0,
                                                   "gen_dim": "{}{}".format(gen, 1 if i.withdim else 0),
                                                   "gen_target_dim": "{}{}".format(gen,
                                                                                   1 if i.y_dim in target_dim else 0),
                                                   "socre_dim": i.fitness.values[0] if i.withdim else 0,
                                                   "socre_target_dim": i.fitness.values[
                                                       0] if i.y_dim in target_dim else 0,
                                                   } for n, i in enumerate(population) if i is not None}
            data_all.update(data)
        random.setstate(rst)

        rst = random.getstate()
        """record"""
        if halloffame is not None:
            halloffame.update(add_ind1)
            if len(halloffame.items) > 0 and halloffame.items[-1].fitness.values[0] >= 0.95:
                print(halloffame.items[-1])
                print(halloffame.items[-1].fitness.values[0])
                break
        random.setstate(rst)

        rst = random.getstate()
        """Dynamic output"""

        record = stats.compile_(population) if stats else {}
        logbook.record(gen=gen, pop=len(population), **record)

        if verbose:
            print(logbook.stream)
        random.setstate(rst)

        """crossover, mutate"""
        offspring = toolbox.select_gs(population, len_pop - elite_size)
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        rst = random.getstate()
        """re-run"""
        offspring.extend(add_ind)
        population[:] = offspring
        random.setstate(rst)

    store = Store()
    store.to_csv(data_all)
    return population, logbook

    #
    # # Begin the generational process
    # random.setstate(rst)
    # for gen in range(1, ngen + 1):
    #     rst = random.getstate()
    #
    #     if store:
    #         rst = random.getstate()
    #         target_dim = toolbox.select_kbest_target_dim.keywords['dim_type']
    #         subp = functools.partial(sub, subed=pset.rep_name_list, subs=pset.real_name_list)
    #         datamnist = {"gen{}_pop{}".format(gen, n): {"gen": gen, "pop": n,
    #                                                "score": i.fitness.values[0],
    #                                                "expr": str(subp(i.expr)),
    #                                                "with_dim": 1 if i.withdim else 0,
    #                                                "dim_is_target_dim": 1 if i.dim in target_dim else 0,
    #                                                "gen_dim": "{}{}".format(gen, 1 if i.withdim else 0),
    #                                                "gen_target_dim": "{}{}".format(gen,
    #                                                                                1 if i.dim in target_dim else 0),
    #                                                "socre_dim": i.fitness.values[0] if i.withdim else 0,
    #                                                "socre_target_dim": i.fitness.values[
    #                                                    0] if i.dim in target_dim else 0,
    #                                                } for n, i in enumerate(population) if i is not None}
    #         data_all.update(datamnist)
    #     random.setstate(rst)
    #     # select_gs the next generation individuals
    #     offspring = toolbox.select_gs(population, len_pop)
    #
    #     # Vary the pool of individuals
    #     offspring = varAnd(offspring, toolbox, cxpb, mutpb)
    #
    #     rst = random.getstate()
    #
    #     # Evaluate the individuals with an invalid fitness
    #     invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    #     # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    #     # fitnesses = parallelize(n_jobs=3, func=toolbox.evaluate, iterable=invalid_ind,  respective=False)
    #     fitnesses = toolbox.parallel(iterable=invalid_ind)
    #     for ind, fit in zip(invalid_ind, fitnesses):
    #         ind.fitness.values = fit[0],
    #         ind.expr = fit[1]
    #         ind.dim = fit[2]
    #         ind.withdim = fit[3]
    #
    #     add_ind = toolbox.select_kbest_target_dim(population, K_best=0.1 * len_pop)
    #     # add_ind2 = toolbox.select_kbest_dimless(population, K_best=0.2 * len_pop)
    #     # add_ind3 = toolbox.select_kbest(population, K_best=5)
    #     # offspring += add_ind
    #     # offspring += add_ind2
    #     # offspring += add_ind3
    #
    #     # Update the hall of fame with the generated individuals
    #     hal = len(add_ind)
    #     if halloffame is not None:
    #         halloffame.update(add_ind)
    #         if len(halloffame.items) > 0 and halloffame.items[-1].fitness.values[0] >= 0.95:
    #             print(halloffame.items[-1])
    #             print(halloffame.items[-1].fitness.values[0])
    #             break
    #
    #     population[:] = offspring
    #     # Append the current generation statistics to the logbook
    #     record = stats.compile(population) if stats else {}
    #     logbook.record(gen=gen, nevals=len(population), **record)
    #     if verbose:
    #         print(logbook.stream)
    #     random.setstate(rst)
    #
    # store = Store()
    # store.to_csv(data_all)
    # return population, logbook


def selKbestDim(pop, K_best=10, dim_type=None, fuzzy=False):
    fit_attr = "fitness"
    chosen = sorted(pop, key=operator.attrgetter(fit_attr))
    chosen.reverse()

    if dim_type is None:
        add_ind = [ind for ind in chosen if ind.withdim == 1]
    elif dim_type is 'integer':
        add_ind = [ind for ind in chosen if ind.y_dim.isinteger]
    elif dim_type is 'ignore':
        add_ind = chosen
    elif isinstance(dim_type, list):
        add_ind = [ind for ind in chosen if ind.y_dim in dim_type]
    elif isinstance(dim_type, Dim):
        if fuzzy:
            add_ind = [ind for ind in chosen if ind.y_dim.is_same_base(dim_type)]
        else:
            add_ind = [ind for ind in chosen if ind.y_dim == dim_type]
    else:
        raise TypeError("dim_type should be None, 'integer', special Dim or list of Dim")
    if K_best is None:
        try:
            K_best = round(len(add_ind) / 10)
        except:
            K_best = 0
    if len(add_ind) >= round(K_best):
        return add_ind[:round(K_best)]
    else:
        return add_ind
