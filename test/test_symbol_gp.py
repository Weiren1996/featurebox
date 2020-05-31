import operator
import time
import unittest

from featurebox.symbol.base import CalculatePrecisionSet
from featurebox.symbol.base import SymbolSet
from featurebox.symbol.base import SymbolTree
from featurebox.symbol.dim import dless
from featurebox.symbol.gp import cxOnePoint, varAnd, genGrow, staticLimit, selRandom, mutShrink, selKbestDim, \
    mutDifferentReplacement
from featurebox.tools.packbox import Toolbox


class MyTestgp(unittest.TestCase):

    def setUp(self):
        self.SymbolTree = SymbolTree
        self.pset = SymbolSet()

        from sklearn.datasets import load_boston

        data = load_boston()
        x = data["data"]
        y = data["target"]

        self.x = x
        self.y = y
        # self.pset.add_features(x, y, )
        self.pset.add_features(x, y, group=[[1, 2], [4, 5]])
        self.pset.add_constants([6, 3, 4], dim=[dless, dless, dless], prob=None)
        self.pset.add_operations(power_categories=(2, 3, 0.5),
                                 categories=("Add", "Mul", "Neg", "Abs"),
                                 self_categories=None)
        self.pset.compress()

        from sklearn.metrics import r2_score, mean_squared_error

        self.cp = CalculatePrecisionSet(self.pset, scoring=[r2_score, mean_squared_error],
                                        score_pen=[1, -1],
                                        filter_warning=True)

    def test_gp_flow(self):
        from numpy import random
        random.seed(1)
        cpset = self.cp
        # def Tree
        from deap.base import Fitness
        from featurebox.tools import newclass
        Fitness_ = newclass.create("Fitness_", Fitness, weights=(1, -1))
        PTree_ = newclass.create("PTrees_", SymbolTree, fitness=Fitness_)

        # def selection
        toolbox = Toolbox()

        # toolbox.register("select", selTournament, tournsize=3)
        toolbox.register("select", selKbestDim, dim_type=dless)
        # selBest
        toolbox.register("mate", cxOnePoint)
        # def mutate
        toolbox.register("generate", genGrow, pset=cpset, min_=2, max_=3)
        # toolbox.register("mutate", mutUniform, expr=toolbox.generate, pset=cpset)
        # toolbox.register("mutate", mutNodeReplacement, pset=cpset)
        toolbox.register("mutate", mutShrink)
        toolbox.register("mutate", mutDifferentReplacement, pset=cpset)

        toolbox.decorate("mate", staticLimit(key=operator.attrgetter("height"), max_value=10))
        toolbox.decorate("mutate", staticLimit(key=operator.attrgetter("height"), max_value=10))
        # def elaluate

        # toolbox.register("evaluate", cpset.parallelize_calculate, n_jobs=4, add_coef=True,
        # inter_add=False, inner_add=False)

        # toolbox.register("parallel", parallelize, n_jobs=1, func=toolbox.evaluate, respective=False, tq=False)

        population = [PTree_.genGrow(cpset, 3, 4) for _ in range(10)]
        # si = sys.getsizeof(cpset)
        for i in range(5):
            xa = time.time()
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            xb = time.time()
            invalid_ind_score = cpset.parallelize_score(inds=invalid_ind, n_jobs=4, batch_size=50)
            x = time.time()
            for ind, score in zip(invalid_ind, invalid_ind_score):
                ind.fitness.values = score[0]
                ind.y_dim = score[1]
            # si2 = sys.getsizeof(invalid_ind[0])
            # invalid_ind=[i.compress() for i in invalid_ind]
            # si3 = sys.getsizeof(invalid_ind[0])
            # print(si3,si2,si)
            a = time.time()
            population = toolbox.select(population, len(population))
            b = time.time()
            offspring = varAnd(population, toolbox, 1, 1)
            c = time.time()
            population[:] = offspring
            d = time.time()
            print("inval", xb - xa, "fuzhi", a - x, "cross_mutate", c - b, "select", b - a, "re", d - c)
            # cpsl.compress()


# if __name__ == '__main__':
#
#     unittest.main()

if __name__ == "__main__":
    import time

    a = time.time()
    se = MyTestgp()
    se.setUp()
    b = time.time()
    se.test_gp_flow()
    c = time.time()
