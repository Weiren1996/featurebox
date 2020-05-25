# -*- coding: utf-8 -*-

# @Time    : 2019/11/12 15:13
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import operator
import random
from functools import partial

import numpy as np
from deap.base import Fitness, Toolbox
from deap.gp import staticLimit, cxOnePoint, mutNodeReplacement, genHalfAndHalf
from deap.tools import HallOfFame, MultiStatistics, Statistics, initIterate, initRepeat, selTournament
from sklearn.metrics import explained_variance_score, r2_score

from featurebox.combination import creator
from featurebox.combination.common import eaSimple, calculatePrecision, selKbestDim
from featurebox.combination.deapbase import ExpressionTree, ExpressionSet
from featurebox.combination.dictbase import FixedSet, FixedTree, generate_index, cxOnePoint_index, mutUniForm_index
from featurebox.combination.dim import Dim, dnan
from featurebox.tools.tool import time_this_function, parallelize

print(DeprecationWarning("this module 'featurebox.combination' would deprecated in version 0.85. "
                         "please turn to 'featurebox.symbol.'"))


@time_this_function
def mainPart(x_, y_, pset, max_=5, pop_n=100, random_seed=2, cxpb=0.8, mutpb=0.1, ngen=5,
             tournsize=3, max_value=10, double=False, score=None, cal_dim=True, target_dim=None,
             inter_add=True, iner_add=True, random_add=False, store=True):
    """

    Parameters
    ----------
    target_dim
    max_
    inter_add
    iner_add
    random_add
    cal_dim
    score
    double
    x_
    y_
    pset
    pop_n
    random_seed
    cxpb
    mutpb
    ngen
    tournsize
    max_value

    Returns
    -------

    """

    if score is None:
        score = [r2_score, explained_variance_score]

    if cal_dim:
        assert all([isinstance(i, Dim) for i in pset.dim_list]), "all import dim of pset should be Dim object"

    random.seed(random_seed)
    toolbox = Toolbox()

    if isinstance(pset, ExpressionSet):
        PTrees = ExpressionTree
        Generate = genHalfAndHalf
        mutate = mutNodeReplacement
        mate = cxOnePoint
    elif isinstance(pset, FixedSet):
        PTrees = FixedTree
        Generate = generate_index
        mutate = mutUniForm_index
        mate = partial(cxOnePoint_index, pset=pset)

    else:
        raise NotImplementedError("get wrong pset")
    if double:
        Fitness_ = creator.create("Fitness_", Fitness, weights=(1.0, 1.0))
    else:
        Fitness_ = creator.create("Fitness_", Fitness, weights=(1.0,))

    PTrees_ = creator.create("PTrees_", PTrees, fitness=Fitness_, dim=dnan, withdim=0)
    toolbox.register("generate", Generate, pset=pset, min_=1, max_=max_)
    toolbox.register("individual", initIterate, container=PTrees_, generator=toolbox.generate)
    toolbox.register('population', initRepeat, container=list, func=toolbox.individual)
    # def selection
    toolbox.register("select_gs", selTournament, tournsize=tournsize)
    toolbox.register("select_kbest_target_dim", selKbestDim, dim_type=target_dim, fuzzy=True)
    toolbox.register("select_kbest_dimless", selKbestDim, dim_type="integer")
    toolbox.register("select_kbest", selKbestDim, dim_type='ignore')
    # def mate
    toolbox.register("mate", mate)
    # def mutate
    toolbox.register("mutate", mutate, pset=pset)
    if isinstance(pset, ExpressionSet):
        toolbox.decorate("mate", staticLimit(key=operator.attrgetter("height"), max_value=max_value))
        toolbox.decorate("mutate", staticLimit(key=operator.attrgetter("height"), max_value=max_value))
    # def elaluate
    toolbox.register("evaluate", calculatePrecision, pset=pset, x=x_, y=y_, scoring=score[0], cal_dim=cal_dim,
                     inter_add=inter_add, iner_add=iner_add, random_add=random_add)
    toolbox.register("evaluate2", calculatePrecision, pset=pset, x=x_, y=y_, scoring=score[1], cal_dim=cal_dim,
                     inter_add=inter_add, iner_add=iner_add, random_add=random_add)
    toolbox.register("parallel", parallelize, n_jobs=1, func=toolbox.evaluate, respective=False)
    toolbox.register("parallel2", parallelize, n_jobs=1, func=toolbox.evaluate2, respective=False)

    pop = toolbox.population(n=pop_n)

    haln = 5
    hof = HallOfFame(haln)

    stats1 = Statistics(lambda ind: ind.fitness.values[0] if ind and ind.y_dim in target_dim else 0)
    stats1.register("max", np.max)

    stats2 = Statistics(lambda ind: ind.y_dim in target_dim if ind else 0)
    stats2.register("countable_number", np.sum)
    stats = MultiStatistics(score1=stats1, score2=stats2)

    population, logbook = eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats,
                                   halloffame=hof, pset=pset, store=store)
    # if not double:
    #     stats1 = Statistics(lambda ind: ind.fitness.values[0])
    #     stats = MultiStatistics(score1=stats1, )
    #     stats.register("avg", np.mean)
    #     stats.register("max", np.max)
    #     population, logbook = eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats,
    #                                halloffame=hof, pset=pset)
    # elif double:
    #     stats1 = Statistics(lambda ind: ind.fitness.values[0])
    #     stats2 = Statistics(lambda ind: ind.fitness.values[1])
    #
    #     stats = MultiStatistics(score1=stats1, score2=stats2)
    #     stats.register("avg", np.mean)
    #     stats.register("max", np.max)
    #     population, logbook = multiEaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, alpha=alpha,
    #                                         halloffame=hof, pset=pset)
    # else:
    #     raise TypeError
    return population, hof
