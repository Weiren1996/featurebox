# #!/usr/bin/python
# # -*- coding: utf-8 -*-
#
# # @Time    : 2019/11/12 15:13
# # @Email   : 986798607@qq.com
# # @Software: PyCharm
# # @License: GNU Lesser General Public License v3.0

import copy
import operator
import os
import time

from deap.base import Fitness
from deap.tools import HallOfFame, Logbook
from numpy import random
from sklearn.metrics import r2_score

from featurebox.symbol.base import CalculatePrecisionSet
from featurebox.symbol.base import SymbolSet
from featurebox.symbol.base import SymbolTree
from featurebox.symbol.functions.dimfunc import Dim
from featurebox.symbol.gp import cxOnePoint, varAnd, genGrow, staticLimit, selKbestDim, \
    selTournament, Statis_func, mutUniform, mutShrink, varAndfus, \
    mutDifferentReplacementVerbose, mutNodeReplacementVerbose, selBest, genFull
from featurebox.tools import newclass
from featurebox.tools.exports import Store
from featurebox.tools.packbox import Toolbox


class BaseLoop(Toolbox):
    """Base loop"""

    def __init__(self, pset, pop=500, gen=20, mutate_prob=0.5, mate_prob=0.8, hall=1, re_hall=None,
                 re_Tree=None, initial_min=None, initial_max=3, max_value=5,
                 scoring=(r2_score,), score_pen=(1,), filter_warning=True, cv=1,
                 add_coef=True, inter_add=True, inner_add=False, vector_add=False,
                 cal_dim=False, dim_type=None, fuzzy=False, n_jobs=1, batch_size=40,
                 random_state=None, stats=None, verbose=True,
                 tq=True, store=False, personal_map=False, stop_condition=None):
        """

        Parameters
        ----------
        pset:SymbolSet
            the feature x and traget y and others should have been added.
        pop:int
            number of popolation
        gen:int
            number of generation
        mutate_prob:float
            probability of mutate
        mate_prob:float
            probability of mate(crossover)
        initial_max:int
            max initial size of expression when first producing.
        initial_min : None,int
            max initial size of expression when first producing.
        max_value:int
            max size of expression
        hall:int,>=1
            number of HallOfFame(elite) to maintain
        re_hall:None or int>=2
            Notes: only vaild when hall
            number of HallOfFame to add to next generation.
        re_Tree: int
            number of new features to add to next generation.
            0 is false to add.
        personal_map:bool or "auto"
            "auto" is using premap and with auto refresh the premap with individual.\n
            True is just using constant premap.\n
            False is just use the prob of terminals.
        scoring: list of Callbale, default is [sklearn.metrics.r2_score,]
            See Also sklearn.metrics
        score_pen: tuple of  1, -1 or float but 0.
            >0 : max problem, best is positive, worse -np.inf
            <0 : min problem, best is negative, worse np.inf
            Notes:
            if multiply score method, the scores must be turn to same dimension in preprocessing
            or weight by score_pen. Because the all the selection are stand on the mean(w_i*score_i)
            Examples: [r2_score] is [1],
        cv=int,sklearn.model_selection._split._BaseKFold
            default =1, means not cv
        filter_warning:bool
            filter warning or not
        add_coef:bool
            add coef in expression or not.
        inter_add：bool
            add intercept constant or not
        inner_add:bool
            dd inner coeffcients or not
        n_jobs:int
            default 1, advise 6
        batch_size:int
            default 40, depend of machine
        random_state:int
            None,int
        cal_dim:bool
            excape the dim calculation
        dim_type:Dim or None or list of Dim
            "coef": af(x)+b. a,b have dimension,f(x) is not dnan. \n
            "integer": af(x)+b. f(x) is interger dimension. \n
            [Dim1,Dim2]: f(x) in list. \n
            Dim: f(x) ~= Dim. (see fuzzy) \n
            Dim: f(x) == Dim. \n
            None: f(x) == pset.y_dim
        fuzzy:bool
            choose the dim with same base with dim_type,such as m,m^2,m^3.
        stats:dict
            details of logbook to show. \n
            Map:\n
            values 
                = {"max": np.max, "mean": np.mean, "min": np.mean, "std": np.std, "sum": np.sum}
            keys
                = {\n
                   "fitness": just see fitness[0], \n
                   "fitness_dim_max": max problem, see fitness with demand dim,\n
                   "fitness_dim_min": min problem, see fitness with demand dim,\n
                   "dim_is_target": demand dim,\n
                   "coef":  dim is true, coef have dim, \n
                   "integer":  dim is integer, \n
                   ...
                   }
            if stats is None, default is :\n
                stats = {"fitness_dim_max": ("max",), "dim_is_target": ("sum",)}   for cal_dim=True
                stats = {"fitness": ("max",)}                                      for cal_dim=False
            if self-definition, the key is func to get attribute of each ind./n
            Examples:
                def func(ind):\n
                    return ind.fitness[0]
                stats = {func: ("mean",), "dim_is_target": ("sum",)}
        verbose:bool
            print verbose logbook or not
        tq:bool
            print progress bar or not
        store:bool or path
            bool or path
        stop_condition:callable
            stop condition on the best ind of hall, which return bool,the true means stop loop.
            Examples:
                def func(ind):\n
                    c = ind.fitness.values[0]>=0.90
                    return c
        """
        super(BaseLoop, self).__init__()
        assert initial_max <= max_value, "the initial size of expression should less than max_value limitation"
        if cal_dim:
            assert all(
                [isinstance(i, Dim) for i in pset.dim_ter_con.values()]), \
                "all import dim of pset should be Dim object."

        random.seed(random_state)

        self.max_value = max_value
        self.pop = pop
        self.gen = gen
        self.mutate_prob = mutate_prob
        self.mate_prob = mate_prob
        self.verbose = verbose
        self.cal_dim = cal_dim
        self.re_hall = re_hall
        self.re_Tree = re_Tree
        self.store = store
        self.data_all = []
        self.personal_map = personal_map
        self.stop_condition = stop_condition

        self.cpset = CalculatePrecisionSet(pset, scoring=scoring, score_pen=score_pen,
                                           filter_warning=filter_warning, cal_dim=cal_dim,
                                           add_coef=add_coef, inter_add=inter_add, inner_add=inner_add,
                                           vector_add=vector_add, cv=cv,
                                           n_jobs=n_jobs, batch_size=batch_size, tq=tq,
                                           fuzzy=fuzzy, dim_type=dim_type,
                                           )

        Fitness_ = newclass.create("Fitness_", Fitness, weights=score_pen)
        self.PTree = newclass.create("PTrees", SymbolTree, fitness=Fitness_)
        # def produce
        if initial_min is None:
            initial_min = 2
        self.register("genGrow", genGrow, pset=self.cpset, min_=initial_min, max_=initial_max,
                      personal_map=self.personal_map)
        self.register("genFull", genFull, pset=self.cpset, min_=initial_min, max_=initial_max,
                      personal_map=self.personal_map)
        self.register("gen_mu", genGrow, min_=1, max_=3, personal_map=self.personal_map)
        # def selection

        self.register("select", selTournament, tournsize=2)

        self.register("selKbestDim", selKbestDim,
                      dim_type=self.cpset.dim_type, fuzzy=self.cpset.fuzzy)
        self.register("selBest", selBest)

        self.register("mate", cxOnePoint)
        # def mutate

        self.register("mutate", mutUniform, expr=self.gen_mu, pset=self.cpset)

        self.decorate("mate", staticLimit(key=operator.attrgetter("height"), max_value=2 * max_value))
        self.decorate("mutate", staticLimit(key=operator.attrgetter("height"), max_value=2 * max_value))

        if stats is None:
            if cal_dim:
                stats = {"fitness_dim_max": ("max",), "dim_is_target": ("sum",)}
            else:
                stats = {"fitness": ("max",)}

        self.stats = Statis_func(stats=stats)
        logbook = Logbook()
        logbook.header = ['gen'] + (self.stats.fields if self.stats else [])
        self.logbook = logbook

        if hall is None:
            hall = 1
        self.hall = HallOfFame(hall)

        if re_hall is None:
            self.re_hall = None
        else:
            if re_hall is 1 or re_hall is 0:
                print("re_hall should more than 1")
                re_hall = 2
            assert re_hall >= hall, "re_hall should more than hall"
            self.re_hall = HallOfFame(re_hall)

    def varAnd(self, *arg, **kwargs):
        return varAnd(*arg, **kwargs)

    def to_csv(self, data_all):
        if self.store:
            if isinstance(self.store, str):
                path = self.store
            else:
                path = os.getcwd()
            file_new_name = "_".join((str(self.pop), str(self.gen),
                                      str(self.mutate_prob), str(self.mate_prob),
                                      str(time.time())))
            try:
                st = Store(path)
                st.to_csv(data_all, file_new_name)
                print("store data to ", path, file_new_name)
            except (IOError, PermissionError):
                st = Store(os.getcwd())
                st.to_csv(data_all, file_new_name)
                print("store data to ", os.getcwd(), file_new_name)

    def maintain_halls(self, population):

        if self.re_hall is not None:
            maxsize = max(self.hall.maxsize, self.re_hall.maxsize)

            if self.cal_dim:
                inds_dim = self.selKbestDim(population, maxsize)
            else:
                inds_dim = self.selBest(population, maxsize)

            self.hall.update(inds_dim)
            self.re_hall.update(inds_dim)

            sole_inds = [i for i in self.re_hall.items if i not in inds_dim]
            inds_dim.extend(sole_inds)
        else:
            if self.cal_dim:
                inds_dim = self.selKbestDim(population, self.hall.maxsize)
            else:
                inds_dim = self.selBest(population, self.hall.maxsize)

            self.hall.update(inds_dim)
            inds_dim = []

        inds_dim = copy.deepcopy(inds_dim)
        return inds_dim

    def re_add(self):
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

    def re_fresh_by_name(self, *arr):
        re_name = ["mutate", "genGrow", "genFull"]
        if len(arr) > 0:
            re_name.extend(arr)
        self.refresh(re_name, pset=self.cpset)

    def run(self):
        # 1.generate###################################################################
        population = [self.PTree(self.genFull()) for _ in range(self.pop)]

        for gen_i in range(1, self.gen + 1):

            population_old = copy.deepcopy(population)

            # 2.evaluate###############################################################
            invalid_ind_score = self.cpset.parallelize_score(population_old)

            for ind, score in zip(population_old, invalid_ind_score):
                ind.fitness.values = tuple(score[0])
                ind.y_dim = score[1]
                ind.dim_score = score[2]

            population = population_old

            # 3.log###################################################################
            # 3.1.log-print##############################

            record = self.stats.compile(population) if self.stats else {}
            self.logbook.record(gen=gen_i, **record)
            if self.verbose:
                print(self.logbook.stream)

            # 3.2.log-store##############################
            if self.store:
                datas = [{"gen": gen_i, "name": str(gen_i), "value": str(gen_i.fitness.values),
                          "dimension": str(gen_i.y_dim),
                          "dim_score": str(gen_i.dim_score)} for gen_i in population]
                self.data_all.extend(datas)

            # 3.3.log-hall###############################
            inds_dim = self.maintain_halls(population)

            # 4.refresh################################################################
            # 4.1.re_update the premap ##################
            if self.personal_map is "auto":
                [self.cpset.premap.update(indi, self.cpset) for indi in inds_dim]

            # 4.2.re_add_tree and refresh pset###########
            if self.re_Tree:
                self.re_add()

            self.re_fresh_by_name()

            # 5.break#######################################################
            if self.stop_condition is not None:
                if self.stop_condition(self.hall.items[0]):
                    break
            # 6.next generation#######################################################
            # selection and mutate,mate
            population = self.select(population, len(population) - len(inds_dim))

            offspring = self.varAnd(population, self, self.mate_prob, self.mutate_prob)

            offspring.extend(inds_dim)
            population[:] = offspring

        # 7.store#####################################################################

        if self.store:
            self.to_csv(self.data_all)
        self.hall.items = [self.cpset.calculate_detail(indi) for indi in self.hall.items]

        return self.hall


class MutilMutateLoop(BaseLoop):
    """
    multiply mutate method.
    """

    def __init__(self, *args, **kwargs):
        """See also BaseLoop"""
        super(MutilMutateLoop, self).__init__(*args, **kwargs)

        self.register("mutate0", mutNodeReplacementVerbose, pset=self.cpset, personal_map=self.personal_map)

        self.register("mutate1", mutUniform, expr=self.gen_mu, pset=self.cpset)
        self.decorate("mutate1", staticLimit(key=operator.attrgetter("height"), max_value=2 * self.max_value))

        self.register("mutate2", mutShrink, pset=self.cpset)

        self.register("mutate3", mutDifferentReplacementVerbose, pset=self.cpset, personal_map=self.personal_map)

    def varAnd(self, population, toolbox, cxpb, mutpb):
        names = self.__dict__.keys()
        import re
        patt = r'mutate[0-9]'
        pattern = re.compile(patt)
        result = [pattern.findall(i) for i in names]
        att_name = []
        for i in result:
            att_name.extend(i)

        self.re_fresh_by_name(*att_name)

        fus = [getattr(self, i) for i in att_name]

        off = varAndfus(population, toolbox, cxpb, mutpb, fus)

        return off


class OnePointMutateLoop(BaseLoop):
    """
    limitation height of population, just use mutNodeReplacementVerbose method.
    """

    def __init__(self, *args, **kwargs):
        """See also BaseLoop"""
        super(OnePointMutateLoop, self).__init__(*args, **kwargs)

        self.register("mutate0", mutNodeReplacementVerbose, pset=self.cpset, personal_map=self.personal_map)

        self.register("mutate3", mutDifferentReplacementVerbose, pset=self.cpset, personal_map=self.personal_map)

    def varAnd(self, population, toolbox, cxpb, mutpb):
        names = self.__dict__.keys()
        import re
        patt = r'mutate[0-9]'
        pattern = re.compile(patt)
        result = [pattern.findall(i) for i in names]
        att_name = []
        for i in result:
            att_name.extend(i)

        self.re_fresh_by_name(*att_name)

        fus = [getattr(self, i) for i in att_name]

        off = varAndfus(population, toolbox, cxpb, mutpb, fus)

        return off


class DimForceLoop(MutilMutateLoop):
    """Force select the individual with target dim for next generation"""

    def __init__(self, *args, **kwargs):
        """See also BaseLoop"""
        super(DimForceLoop, self).__init__(*args, **kwargs)
        assert self.cal_dim == True, "For DimForceLoop type, the 'cal_dim' must be True"

        self.register("select", selKbestDim,
                      dim_type=self.cpset.dim_type, fuzzy=self.cpset.fuzzy, force_number=True)

# if __name__ == "__main__":
#     # data
#     data = load_boston()
#     x = data["data"]
#     y = data["target"]
#     c = [6, 3, 4]
#     # unit
#     from sympy.physics.units import kg
#
#     x_u = [kg] * 13
#     y_u = kg
#     c_u = [dless, dless, dless]
#
#     x, x_dim = Dim.convert_x(x, x_u, target_units=None, unit_system="SI")
#     y, y_dim = Dim.convert_xi(y, y_u)
#     c, c_dim = Dim.convert_x(c, c_u)
#
#     z = time.time()
#
#     # symbolset
#     pset0 = SymbolSet()
#     pset0.add_features(x, y, x_dim=x_dim, y_dim=y_dim, x_group=[[1, 2], [3, 4], [5, 6]])
#     pset0.add_constants(c, c_dim=c_dim, c_prob=None)
#     pset0.add_operations(power_categories=(2, 3, 0.5),
#                          categories=("Add", "Mul", "Sub", "Div", "exp", "Abs"))
#
#     # a = time.time()
#     bl = MutilMutateLoop(pset=pset0, gen=4, pop=10, hall=2, batch_size=40, re_hall=2,
#                          n_jobs=1, mate_prob=1, max_value=10, initial_max=3,
#                          mutate_prob=0.8, tq=True, dim_type="coef",
#                          re_Tree=2, store=False, random_state=1,
#                          stats={"fitness_dim_max": ["max"], "dim_is_target": ["sum"], "height": ["mean"]},
#                          add_coef=True, cal_dim=True, inner_add=False, vector_add=True, personal_map=False)
#     # b = time.time()
#     bl.run()
#     population = [bl.PTree(bl.genFull()) for _ in range(30)]
#     pset = bl.cpset
#     for i in population:
#         # i.ppprint(bl.cpset)
#         # i = "exp(gx0/gx1)"
#
#         i = compile_context(i, pset.context, pset.gro_ter_con, simplify=False)
#         # print(i)
#         # print(i)
#         # fun = Coef("V", np.array([1.4,1.3]))
#         # i = fun(i)
#         # f = Function("MAdd")
#         # i = f(i)
#         try:
#             # group_str(i,pset)
#             # i=general_expr(i, pset, simplifying=True)
#             i = general_expr(i, pset, simplifying=False)
#             # print(i)
#             # print(i)
#             # pprint(i)
#         except NotImplementedError as e:
#             print(e)
#     # c = time.time()
#     # print(c - b, b - a, a - z)
#     a, b, c = sympy.Symbol("a"), sympy.Symbol("b"), sympy.Symbol("c")
#     print(sympy.simplify((a + (b + 1)) * c) == sympy.simplify(a * c + b * c + c))
