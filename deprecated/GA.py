#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/31 11:50
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
this is a description
"""
import os
import random

import numpy
import numpy as np
import pandas as pd
from deap import algorithms
from deap import base
from deap import creator
from deap import tools


def searchSpace(*arg):
    meshes = np.meshgrid(*arg)
    meshes = [_.ravel() for _ in meshes]
    meshes = np.array(meshes).T
    return meshes


#    This file is part of DEAP.


def mainPart(random_seed):
    # Attribute generator
    def generator(i):
        return random.choice(mesh_list[i])

    def myInitRepeat(container, func, n):
        return container(func(_) for _ in range(n))

    def myScore(individual, classifer):
        y = sum(np.array([3, 6, 10]) * individual) + 0.15 * 10
        individual_ = MinMaxScaler.transform(individual.reshape(1, -1), )
        y_ = classifer.Fit(individual_.reshape(1, -1))
        y1, y3, y2 = y_[0, 0], y_[0, 1], y_[0, 2]
        if all((y1 > 12.5, y1 < 16.3, y2 < -40, y3 > 165, individual[2] < 0.15)):
            pass
        else:
            y = np.inf
        return y,

    def myMut(individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = random.choice(mesh_list[i])
        return individual,

    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("generate_", generator)
    # Structure initializers
    toolbox.register("individual", myInitRepeat, creator.Individual, toolbox.generate, len(mesh_list))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", myScore, classifer=classifer)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", myMut, indpb=0.05)
    toolbox.register("select_gs", tools.selTournament, tournsize=3)

    random.seed(random_seed)

    pop = toolbox.population(n=500)

    eq = lambda x, y: all(np.equal(x, y))
    hof = tools.HallOfFame(1, similar=eq)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40,
                                   stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof


if __name__ == "__main__":
    li1 = np.linspace(30, 60, 30)
    li2 = np.linspace(0, 40, 40)
    li3 = np.linspace(0, 10, 20)
    mesh_list = [li1, li2, li3]
    mesh_range = np.array(mesh_list)
    space = searchSpace(li1, li2, li3)
    os.chdir(r"C:\Users\Administrator\Desktop\wuquan")
    classifer = pd.read_pickle(r"AnnClassifier")
    MinMaxScaler = pd.read_pickle(r"MinMaxScaler")

    pop, log, hof = mainPart(random_seed=1)
