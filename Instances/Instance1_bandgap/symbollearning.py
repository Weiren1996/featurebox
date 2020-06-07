import time

from sklearn import preprocessing
from sklearn.datasets import load_boston
from sklearn.utils import shuffle

from featurebox.symbol.base import SymbolSet
from featurebox.symbol.dim import dless, Dim
from featurebox.symbol.flow import BaseLoop, DimForceLoop, MutilMutateLoop
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

    # from sympy.physics.units import eV
    # select = ['electronegativity(martynov&batsanov)', 'fusion enthalpy', 'valence electron number']
    # select_unit = [dless, eV, dless]

    # from sympy.physics.units import eV,pm
    # select = ['covalent radii', 'electronegativity(martynov&batsanov)', 'valence electron number']
    # select_unit = [pm, dless, dless]

    # from sympy.physics.units import eV,pm
    # select = ['electronegativity(martynov&batsanov)', 'ionic radii(shannon)', 'valence electron number']
    # select_unit = [dless,pm, dless]

    # from sympy.physics.units import eV,pm
    # select = ['covalent radii', 'valence electron number']
    # select_unit = [pm, dless]

    # from sympy.physics.units import eV,pm
    # select = ['covalent radii', 'fusion enthalpy', 'valence electron number']
    # select_unit = [pm, eV,dless]

    # from sympy.physics.units import eV,pm
    # select = ['electronegativity(martynov&batsanov)', 'valence electron number']
    # select_unit = [dless,dless]

    # from sympy.physics.units import eV,pm
    # select = ['covalent radii', 'ionic radii(shannon)', 'valence electron number']
    # select_unit = [pm,pm,dless]

    # from sympy.physics.units import eV,pm
    # select = ['core electron distance(schubert)', 'covalent radii', 'valence electron number']
    # select_unit = [pm,pm,dless]

    # from sympy.physics.units import eV,pm
    # select = ['core electron distance(schubert)', 'electronegativity(martynov&batsanov)', 'valence electron number']
    # select_unit = [pm,dless,dless] ###

    from sympy.physics.units import eV,pm
    select = ['cohesive energy(Brewer)', 'covalent radii', 'valence electron number']
    select_unit = [eV,pm,dless]


    select = [j + "_%i" % i for j in select for i in range(2)]
    x_u = [j for j in select_unit for i in range(2)]

    X_frame = data225_import[select]
    y_frame = data225_import['exp_gap']
    X = X_frame.values
    y = y_frame.values
    x, y = shuffle(X, y, random_state=5)

    # y_unit
    from sympy.physics.units import eV, elementary_charge, m

    y_u = eV
    # c_unit
    c = [1, 5.290 * 10 ** -11, 1.74, 2, 3, 4, 0.5, 1 / 3, 1 / 4]
    c_u = [elementary_charge, m, dless,dless,dless,dless,dless,dless,dless]

    """preprocessing"""
    x, x_dim = Dim.convert_x(x, x_u, target_units=None, unit_system="SI")
    y, y_dim = Dim.convert_xi(y, y_u)
    c, c_dim = Dim.convert_x(c, c_u)

    scal = MagnitudeTransformer(tolerate=1)
    x, y = scal.fit_transform_all(x, y, group=2)
    c = scal.fit_transform_constant(c)

    # symbolset
    pset0 = SymbolSet()
    pset0.add_features(x, y, x_dim=x_dim, y_dim=y_dim, group=2)
    pset0.add_constants(c, dim=c_dim, prob=0.05)
    pset0.add_operations(power_categories=(2, 3, 0.5),
                         categories=("Add", "Mul", "Sub", "Div", "exp"),
                         self_categories=None)

    # a = time.time()
    bl = MutilMutateLoop(pset=pset0, gen=10, pop=500, hall=1, batch_size=40, re_hall=3,
                         n_jobs=6, mate_prob=0.8, max_value=5,
                         mutate_prob=0.5, tq=True, dim_type="coef",
                         re_Tree=0, store=False, random_state=0,
                         stats={"fitness_dim_max": ["max"], "dim_is_target": ["sum"]},
                         add_coef=True, inner_add=False, cal_dim=True, personal_map=False)
    # b = time.time()
    exps = bl.run()
    print([i.coef_expr for i in exps])
    # c = time.time()
    # print(c - b, b - a, a - z)
