from collections import deque

import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap.algorithms import varAnd
from deap.tools import mutShuffleIndexes
from sklearn.datasets import load_boston
from sklearn.svm import SVR


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

    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    best = deque([halloffame.items[0].fitness.values[0]], maxlen=15)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Append the current generation statistics to the logbook
        record = stats.compile_(population + halloffame.items) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - halloffame.maxsize)
        offspring.extend(halloffame.items)
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        best.append(halloffame.items[0].fitness.values[0])
        if sum(best) / best.maxlen == best[-1]:
            break
        # Replace the current population by the offspring
        population[:] = offspring

    return population, logbook


def generate_xi():
    return np.random.randint(0, 2)


def generate(space):
    return [generate_xi() for i in range(space)]


def GA(x, y, fit_func, pop_n=1000, hof_n=1, cxpb=0.6, mutpb=0.3, ngen=40, max_or_min="max", mut_indpb=0.05):
    x_space = x.shape[1]
    if max_or_min == "max":
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    else:
        creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("generate_x", generate, x_space)
    toolbox.register("generate_xi", generate_xi, x_space)
    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.generate_x)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fit_func, x=x, y=y)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutShuffleIndexes, indpb=mut_indpb)
    toolbox.register("select", tools.selTournament, tournsize=3)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop = toolbox.population(n=pop_n)
    hof = tools.HallOfFame(hof_n)

    eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
             stats=stats, halloffame=hof, verbose=True)
    for i in hof.items:
        print(i, i.fitness)
    return hof.items


if __name__ == "__main__":
    data = load_boston()
    x = data.data
    y = data.target
    svr = SVR(gamma="scale")


    def fitn(ind, x, y):
        index = np.where(np.array(ind) == 1)[0]
        x = x[:, index]
        if x.shape[1] > 1:

            svr = SVR(gamma="scale")
            svr.fit(x, y)
            sc = svr.score(x, y)

            return sc,
        else:
            return 0,


    best = GA(x, y, fitn, pop_n=500, hof_n=1, cxpb=0.8, mutpb=0.4, ngen=10, max_or_min="max", mut_indpb=0.1)
