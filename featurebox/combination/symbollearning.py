#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/30 18:10
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
this is a description
"""
import operator
import random
import warnings
from functools import partial
import numpy as np
import pandas as pd
from deap import gp
from deap.base import Fitness, Toolbox
from deap.gp import PrimitiveSet, cxOnePoint, mutNodeReplacement
from deap.tools import HallOfFame, MultiStatistics, Statistics, initIterate, initRepeat, selTournament
from sklearn.metrics import r2_score
from featurebox.combination.symbolbase import ExpressionTree, FixedExpressionTree, FixedPrimitiveSet, \
    calculate, create, \
    cxOnePoint_index, generate, get_name, multiEaSimple, mutUniForm_index, sympyPrimitiveSet
from featurebox.tools.exports import Store
from featurebox.tools.imports import Call
from featurebox.tools.tool import time_this_function

warnings.filterwarnings("ignore")


@time_this_function
def main_part(x_, y_, pset, pop_n=100, random_seed=1, cxpb=0.8, mutpb=0.1, ngen=5, max_=None, alpha=1,
              tournsize=3, max_value=10, **kargs):
    """

    Parameters
    ----------
    x_
    y_
    pset
    pop_n
    random_seed
    cxpb
    mutpb
    ngen
    max_
    alpha
    tournsize
    max_value
    kargs

    Returns
    -------

    """
    random.seed(random_seed)
    toolbox = Toolbox()
    if isinstance(pset, PrimitiveSet):
        PTrees = ExpressionTree
        Generate = gp.genHalfAndHalf
        mutate = mutNodeReplacement
        mate = cxOnePoint
    elif isinstance(pset, FixedPrimitiveSet):
        PTrees = FixedExpressionTree
        Generate = generate
        mate = partial(cxOnePoint_index, pset=pset)
        mutate = mutUniForm_index
    else:
        raise NotImplementedError("get wrong pset")

    Fitness_ = create("Fitness_", Fitness, weights=(1.0, 1.0))
    PTrees_ = create("PTrees_", PTrees, fitness=Fitness_)
    toolbox.register("generate", Generate, pset=pset, min_=2, max_=max_)
    toolbox.register("individual", initIterate, container=PTrees_, generator=toolbox.generate)
    toolbox.register('population', initRepeat, container=list, func=toolbox.individual)
    # def selection
    toolbox.register("select", selTournament, tournsize=tournsize)
    # def mate
    toolbox.register("mate", mate)
    # def mutate
    toolbox.register("mutate", mutate, pset=pset)
    if isinstance(pset, PrimitiveSet):
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_value))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_value))
    # def elaluate
    toolbox.register("evaluate", calculate, pset=pset, x=x_, y=y_, score_method=r2_score, **kargs)
    toolbox.register("evaluate2", calculate, pset=pset, x=x_, y=y_, score_method=r2_score, **kargs)

    stats1 = Statistics(lambda ind: ind.fitness.values[0])
    stats = MultiStatistics(score1=stats1, )
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop = toolbox.population(n=pop_n)
    haln = 5
    hof = HallOfFame(haln)

    population, logbook = multiEaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, alpha=alpha,
                                        halloffame=hof, pset=pset)

    return population, logbook, hof


if __name__ == "__main__":
    store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp_last')
    data = Call(r'C:\Users\Administrator\Desktop\band_gap_exp_last')
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
    # x, y = utils.shuffle(x, y, random=5)
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
    pset1 = sympyPrimitiveSet(rep_name=rep_name, types=None,
                              categories=('Add', 'Sub', 'Mul', 'Div', 'Max', "Rec", 'exp', "log", "Abs"),
                              power_categories=(1 / 3, 1 / 2, 1, 2, 3))

    result = main_part(x, y, pset1, pop_n=500, random_seed=1, cxpb=0.8, mutpb=0.1, ngen=10, max_=2,
                       inter_add=True, iner_add=False, random_add=False)
