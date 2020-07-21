import numpy as np
from sklearn.utils import shuffle

from featurebox.symbol.base import SymbolSet
from featurebox.symbol.calculation.translate import general_expr_dict, group_str, general_expr
from featurebox.symbol.flow import MutilMutateLoop
from featurebox.symbol.functions.dimfunc import Dim, dless
from featurebox.symbol.preprocess import MagnitudeTransformer
from featurebox.tools.exports import Store
from featurebox.tools.imports import Call

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
    c = [1, 5.290 * 10 ** -11, 1.74, 2, 3, 4, 1 / 2, 1 / 3, 1 / 4]
    c_u = [elementary_charge, m, dless, dless, dless, dless, dless, dless, dless]

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
    pset0.add_operations(power_categories=(2, 3, 0.5, 1/3),
                         categories=("Add", "Mul", "Sub", "Div", "exp", "ln"),
                         self_categories=None)

    total_height = 4

    # stop = None

    # stop = lambda ind: ind.fitness.values[0] >= 0.95
    # bl = MutilMutateLoop(pset=pset0, gen=20, pop=1000, hall=1, batch_size=40, re_hall=3,
    #                      n_jobs=12, mate_prob=0.9, max_value=7, initial_min=2, initial_max=4,
    #                      mutate_prob=0.8, tq=False, dim_type="coef", stop_condition=stop,
    #                      re_Tree=0, store=False, random_state=1, verbose=True,
    #                      stats={"fitness_dim_max": ["max"], "dim_is_target": ["sum"], "h_GVP": ["mean"]},
    #                      add_coef=True, inter_add=True, inner_add=True, cal_dim=True, vector_add=True,
    #                      personal_map=False)
    # pset = bl.cpset
    #
    # bl.run()
    # ind = bl.hall.items[0]
    # expr = ind.coef_expr
    # exprr = general_expr(expr, pset, simplifying=True)


    stop = lambda ind: ind.fitness.values[0] >= 0.99
    bl = MutilMutateLoop(pset=pset0, gen=20, pop=1000, hall=1, batch_size=40, re_hall=3,
                         n_jobs=12, mate_prob=0.9, max_value=7, initial_min=2, initial_max=4,
                         mutate_prob=0.8, tq=False, dim_type="coef", stop_condition=stop,
                         re_Tree=0, store=False, random_state=1, verbose=True,
                         stats={"fitness_dim_max": ["max"], "dim_is_target": ["sum"], "h_GVP": ["mean"]},
                         add_coef=True, inter_add=True, inner_add=True, cal_dim=False, vector_add=True,
                         personal_map=False)
    pset = bl.cpset

    bl.run()
    ind = bl.hall.items[0]
    expr = ind.coef_expr


