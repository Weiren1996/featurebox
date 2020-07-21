from itertools import product, chain

import numpy as np
from sklearn.utils import shuffle

from featurebox.symbol.base import SymbolSet, SymbolTree
from featurebox.symbol.calculation.translate import general_expr_dict, compile_context
from featurebox.symbol.flow import OnePointMutateLoop
from featurebox.symbol.functions.dimfunc import Dim, dless
from featurebox.symbol.preprocess import MagnitudeTransformer
from featurebox.tools.exports import Store
from featurebox.tools.imports import Call
from featurebox.tools.tool import tt

if __name__ == "__main__":
    import os

    os.chdir(r'band_gap')
    data = Call()
    all_import = data.csv().all_import
    name_and_abbr = data.csv().name_and_abbr

    store = Store()

    data_import = all_import
    data225_import = data_import

    select = ['cell volume', 'cell density',
              'lattice constants a', 'lattice constants c', 'covalent radii', 'ionic radii(shannon)',
              'core electron distance(schubert)',
              'fusion enthalpy', 'cohesive energy(Brewer)', 'total energy',
              'effective nuclear charge(slater)', 'valence electron number', 'electronegativity(martynov&batsanov)',
              'atomic volume(villars,daams)']
    from sympy.physics.units import eV, pm, nm

    select_unit = [100 ** 3 * pm ** 3, 100 ** -3 * pm ** -3, 100 * pm, 100 * pm, 100 * pm, 100 * pm, 100 * pm, eV, eV,
                   eV, dless, dless, dless, 10 ** -2 * nm ** 3]

    fea_name = ['V_c', 'rho_c'] + [name_and_abbr[j][1] + "_%i" % i for j in select[2:] for i in range(2)]
    select = ['cell volume', 'cell density'] + [j + "_%i" % i for j in select[2:] for i in range(2)]
    x_u = [100 ** 3 * pm ** 3, 100 ** -3 * pm ** -3] + [j for j in select_unit[2:] for i in range(2)]

    X_frame = data225_import[select]
    y_frame = data225_import['exp_gap']
    X = X_frame.values
    y = y_frame.values
    x, y = shuffle(X, y, random_state=5)

    # y_unit
    from sympy.physics.units import eV, elementary_charge, m, pm

    y_u = eV
    # c_unit
    c = [1, 5.290 * 10 ** -11, 1.74]
    c_u = [elementary_charge, m, dless]

    """preprocessing"""
    x, x_dim = Dim.convert_x(x, x_u, target_units=None, unit_system="SI")
    y, y_dim = Dim.convert_xi(y, y_u)
    c, c_dim = Dim.convert_x(c, c_u)

    scal = MagnitudeTransformer(tolerate=1)
    group = 2
    n = X.shape[1]
    indexes = [_ for _ in range(n)]
    group = [indexes[i:i + group] for i in range(2, len(indexes), group)]
    x, y = scal.fit_transform_all(x, y, group=group)
    c = scal.fit_transform_constant(c)

    # symbolset
    pset0 = SymbolSet()
    x_g = np.arange(x.shape[1])
    x_g = x_g.reshape(-1, 2)
    x_g = list(x_g[1:])
    pset0.add_features(x, y, x_dim=x_dim, y_dim=y_dim, x_group=x_g, feature_name=fea_name)
    pset0.add_constants(c, c_dim=c_dim, c_prob=0.05)
    pset0.add_operations(power_categories=(2, 3, 0.5),
                         categories=("Add", "Mul", "Sub", "Div", "exp", "ln"),
                         self_categories=None)

    height = 2
    h_GVP = 1

    # stop = None
    stop = lambda ind: ind.fitness.values[0] >= 0.880963
    bl = OnePointMutateLoop(pset=pset0, gen=10, pop=1000, hall=1, batch_size=40, re_hall=3,
                            n_jobs=12, mate_prob=0.9, max_value=1, initial_min=1, initial_max=1,
                            mutate_prob=0.8, tq=True, dim_type="coef", stop_condition=stop,
                            re_Tree=0, store=False, random_state=8, verbose=True,
                            stats={"fitness_dim_max": ["max"], "dim_is_target": ["sum"]},
                            add_coef=True, inter_add=True, inner_add=False, cal_dim=True, vector_add=False,
                            personal_map=False)
    pset = bl.cpset
    """exhaustion show all under 2, get the best """
    def find_best():
        prim = bl.cpset.primitives

        ter = bl.cpset.terminals_and_constants
        prim1 = [_ for _ in prim if _.arity==1]
        prim2 = [_ for _ in prim if _.arity==2]
        dispose = bl.cpset.dispose

        top = [i for i in dispose if i.name not in ["Self","Conv"]]
        pop_all2 = product(top,prim2,dispose,ter,dispose,ter)

        pop_all1 = product(top,prim1,dispose,ter)
        pop_all = chain(pop_all1,pop_all2)
        pop_all = list(pop_all)
        tt.t1
        pop_all = [SymbolTree(i) for i in pop_all]
        tt.t2
        invalid_ind_score = bl.cpset.parallelize_score(pop_all)
        tt.t3
        score = [(i[0]*i[2]) for i in invalid_ind_score]
        tt.t4
        score = np.array(score)
        index = np.argmax(score)
        tt.t5
        tt.p
        score_best = score[index]
        pop_all_best = pop_all[int(index)]

        i = compile_context(pop_all_best, pset.context, pset.gro_ter_con, simplify=True)
        expr = general_expr_dict(i, pset.expr_init_map, pset.free_symbol,
                                 pset.gsym_map, simplifying=True)

        return len(pop_all), expr
    # a = find_best()
    """end"""

    bl.run()
    ind = bl.hall.items[0]
    expr = ind.expr
