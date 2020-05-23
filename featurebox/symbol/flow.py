# #!/usr/bin/python
# # -*- coding: utf-8 -*-
#
# # @Time    : 2019/11/12 15:13
# # @Email   : 986798607@qq.com
# # @Software: PyCharm
# # @License: BSD 3-Clause

import copy
import operator
import os

from deap.base import Fitness
from deap.tools import HallOfFame, Logbook
from numpy import random
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

from featurebox.symbol.base import CalculatePrecisionSet
from featurebox.symbol.base import SymbolSet
from featurebox.symbol.base import SymbolTree
from featurebox.symbol.dim import dless, Dim, dnan
from featurebox.symbol.gp import cxOnePoint, varAnd, genGrow, staticLimit, selKbestDim, \
    selTournament, Statis_func, mutNodeReplacement
from featurebox.tools import newclass
from featurebox.tools.exports import Store
from featurebox.tools.packbox import Toolbox


class BaseLoop(Toolbox):
    """base loop"""

    def __init__(self, pset, pop=500, gen=20, mutate_prob=0.1, mate_prob=0.5,
                 hall=1, re_hall=3, re_Tree=1, initial_max=3, max_value=10,
                 scoring=(r2_score,), score_pen=(1,), filter_warning=True,
                 add_coef=True, inter_add=True, inner_add=False,
                 cal_dim=True, dim_type=None, fuzzy=False,
                 n_jobs=1, batch_size=10, random_state=None,
                 stats=None, verbose=True, tq=True, store=True
                 ):
        """

        Parameters
        ----------
        pset:SymbolSet
            the feature x and traget y and others should have been added.
        pop:int
            popolation
        gen:int
            number of generation
        mutate_prob:float
            probability of mutate
        mate_prob:float
            probability of mate(crossover)
        initial_max:int
            max initial size of expression when first producing.
        max_value:int
            max size of expression
        hall:int
            number of HallOfFame(elite) to store
        re_hall: None or int
            Notes: must >=2
            number of HallOfFame to add to next generation.
        re_Tree: int
            number of new features to add to next generation.
            0 is false to add.
        scoring: list of Callbale, default is [sklearn.metrics.r2_score,]
            See Also sklearn.metrics
        score_pen: tuple of  1, -1 or float but 0.
            >0 : best is positive, worse -np.inf
            <0 : best is negative, worse np.inf
            Notes:
            if multiply score method, the scores must be turn to same dimension in preprocessing
            or weight by score_pen. Because the all the selection are stand on the mean(w_i*score_i)
            Examples: [r2_score] is [1],
        filter_warning:bool
        add_coef:bool
        inter_add：bool
        inner_add:bool
        n_jobs:int
        batch_size:int
        random_state:int
        cal_dim:bool
        dim_type:Dim or None or list of Dim
        fuzzy:bool
        stats:bool
        verbose:bool
        tq:bool
        store:bool
        """

        if cal_dim:
            assert all(
                [isinstance(i, Dim) for i in pset.dim_ter_con.values()]), \
                "all import dim of pset should be Dim object."

        random.seed(random_state)
        pset.compress()
        self.cpset = CalculatePrecisionSet(pset, scoring=scoring, score_pen=score_pen,
                                           filter_warning=filter_warning, cal_dim=cal_dim,
                                           add_coef=add_coef, inter_add=inter_add, inner_add=inner_add,
                                           n_jobs=n_jobs, batch_size=batch_size, tq=tq)

        Fitness_ = newclass.create("Fitness_", Fitness, weights=score_pen)
        self.PTree = newclass.create("PTrees", SymbolTree, fitness=Fitness_)
        # def produce
        self.register("genGrow", genGrow, pset=self.cpset, min_=2, max_=initial_max)
        # def selection

        self.register("select", selTournament, tournsize=3)

        dim_type = self.cpset.y_dim if not dim_type else dim_type
        self.register("selKbestDim", selKbestDim, dim_type=dim_type, fuzzy=fuzzy)
        # selBest
        self.register("mate", cxOnePoint)
        # def mutate
        self.register("gen_mu", genGrow, pset=self.cpset, min_=2, max_=3)

        # self.register("mutate", mutUniform, expr=self.gen_mu, pset=self.cpset)
        self.register("mutate", mutNodeReplacement, pset=self.cpset)
        # self.register("mutate", mutShrink)
        # self.register("mutate", mutDifferentReplacement, pset=self.cpset)

        self.decorate("mate", staticLimit(key=operator.attrgetter("height"), max_value=max_value))
        self.decorate("mutate", staticLimit(key=operator.attrgetter("height"), max_value=max_value))
        self.stats = Statis_func(stats=stats)
        logbook = Logbook()
        logbook.header = ['gen'] + (self.stats.fields if self.stats else [])
        self.logbook = logbook

        self.hall = HallOfFame(hall)
        self.pop = pop
        self.gen = gen
        self.mutate_prob = mutate_prob
        self.mate_prob = mate_prob
        self.verbose = verbose
        self.cal_dim = cal_dim
        self.re_hall = re_hall
        self.re_Tree = re_Tree
        self.store = store

    def run(self):

        population = [self.PTree(self.genGrow()) for _ in range(self.pop)]
        data_all = []
        for gen_i in range(self.gen):
            # evaluate################################################################

            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            invalid_ind_score = self.cpset.parallelize_score(invalid_ind)

            for ind, score in zip(invalid_ind, invalid_ind_score):
                ind.fitness.values = score[0]
                ind.y_dim = score[1]
                ind.dim_score = score[2]

            # hall###################################################################
            if self.re_hall:
                if self.cal_dim:
                    inds_dim = self.selKbestDim(population, self.re_hall)
                else:
                    inds_dim = self.select(population, self.re_hall)
            else:
                inds_dim = []

            if self.hall is not None:
                self.hall.update(inds_dim)
            record = self.stats.compile(population) if self.stats else {}
            self.logbook.record(gen=gen_i, **record)
            if self.verbose:
                print(self.logbook.stream)
            if self.store:
                datas = [{"gen": gen_i, "name": str(gen_i), "value": str(gen_i.fitness.values),
                          "dimension": str(gen_i.y_dim),
                          "target_dim": str(gen_i.dim_score)} for gen_i in population]
                data_all.extend(datas)
            # next generation
            # selection and mutate,mate##############################################

            population = self.select(population, len(population) - len(inds_dim))
            offspring = varAnd(population, self, self.mate_prob, self.mutate_prob)
            offspring.extend(inds_dim)
            population[:] = offspring

            # re_tree################################################################
            if self.hall.items and self.re_Tree:
                it = self.hall.items
                indo = it[random.choice(len(it))]
                ind = copy.deepcopy(indo)
                inds = ind.depart()
                if not inds:
                    pass
                else:
                    inds = [self.cpset.calculate_detail(indi) for indi in inds]
                    le = min(self.re_Tree, len(inds))
                    indi = inds[random.choice(le)]
                    self.cpset.add_tree_to_features(indi)
                    self.refresh(("gen_mu", "genGrow"), pset=self.cpset)
        if self.store:
            st = Store(os.getcwd())
            st.to_csv(data_all.T)
            print("store data to ", os.getcwd())


if __name__ == "__main__":
    pset0 = SymbolSet()
    data = load_boston()
    x = data["data"]
    y = data["target"]

    # self.pset.add_features(x, y, )
    pset0.add_features(x, y, group=[[1, 2], [4, 5]])
    pset0.add_constants([6, 3, 4], dim=[dless, dless, dnan], prob=None)
    pset0.add_operations(power_categories=(2, 3, 0.5),
                         categories=("Add", "Mul", "Neg", "Abs"),
                         self_categories=None)
    bl = BaseLoop(pset=pset0, gen=8, pop=500, hall=2, batch_size=50, n_jobs=10, re_Tree=0, store=False)
    bl.run()
