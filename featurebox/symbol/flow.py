#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Time    : 2019/11/12 15:13
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

from numpy import random

import operator
from functools import partial

import numpy as np
from deap.base import Fitness, Toolbox
from deap.gp import staticLimit
from deap.tools import Logbook, HallOfFame, MultiStatistics, Statistics
from sklearn.metrics import explained_variance_score, r2_score

from featurebox.symbol.base import SymbolSet, SymbolTree, CalculatePrecisionSet
from featurebox.symbol.dim import Dim
from featurebox.symbol.gp import selTournament, selKbestDim, mutUniform, generate, cxOnePoint, varAnd
from featurebox.tools import newclass
from featurebox.tools.tool import time_this_function, parallelize


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = Logbook()
    logbook.header = ['gen'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    invalid_ind = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind in invalid_ind:
        ind.fitness.values = ind.coef_score

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        invalid_ind = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind in invalid_ind:
            ind.fitness.values = ind.coef_score
            # ind.compress()

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


@time_this_function
def flow(pset, max_=5, pop_n=100, random_seed=2, cxpb=0.8, mutpb=0.1, ngen=5,
         tournsize=3, max_value=10, scoring=None, cal_dim=True,
         add_coef=True, inter_add=True, inner_add=True, store=True, score_pen=None,
         stats=None):

    cpset = CalculatePrecisionSet(pset, scoring=scoring, score_pen=score_pen, filter_warning=True, cal_dim=cal_dim)

    if cal_dim:
        assert all(
            [isinstance(i, Dim) for i in cpset.dim_ter_con.value()]), "all import dim of pset should be Dim object"

    np.random.seed(random_seed)

    # def Tree
    Fitness_ = newclass.create("Fitness_", Fitness, weights=score_pen)
    PTree = newclass.create("PTrees_", SymbolTree, fitness=Fitness_)

    # def selection
    toolbox = Toolbox()

    toolbox.register("select", selTournament, tournsize=tournsize)
    toolbox.register("select_k_best_target_dim", selKbestDim, dim_type=cpset.y_dim, fuzzy=False)
    toolbox.register("select_k_best_dimless", selKbestDim, dim_type="integer")
    toolbox.register("select_k_best", selKbestDim, dim_type='ignore')
    # def mate
    toolbox.register("mate", cxOnePoint)
    # def mutate
    toolbox.register("generate", generate, pset=cpset, min_=1, max_=max_)
    toolbox.register("mutate", mutUniform, expr=toolbox.generate, pset=cpset)

    toolbox.decorate("mate", staticLimit(key=operator.attrgetter("height"), max_value=max_value))
    toolbox.decorate("mutate", staticLimit(key=operator.attrgetter("height"), max_value=max_value))
    # def elaluate
    toolbox.register("evaluate", cpset.calculate_detail, add_coef=add_coef, inter_add=inter_add, inner_add=inner_add)
    toolbox.register("parallel", parallelize, n_jobs=1, func=toolbox.evaluate, respective=False, tq=True)

    pop = [PTree.genGrow(cpset, min_=2, max_=max_) for _ in range(pop_n)]

    haln = 5
    hof = HallOfFame(haln)

    sa_all = {}

    if stats:
        for i, si in enumerate(stats):
            sa = Statistics(si[0])
            sa.register(si[1], si[2])
            sa_all["Calculate%s" % i] = sa
            stats = MultiStatistics(sa_all)

    population, logbook = eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats,
                                   halloffame=hof, pset=cpset, store=store)

    return hof

    # stats1 = Statistics(lambda ind: ind.fitness.values[0] if ind and ind.y_dim in target_dim else 0)
    # stats1.register("max", np.max)
    #
    # stats2 = Statistics(lambda ind: ind.y_dim in target_dim if ind else 0)
    # stats2.register("countable_number", np.sum)
