import time
from sklearn.datasets import load_boston
from sklearn.utils import shuffle

from featurebox.symbol.base import SymbolSet
from featurebox.symbol.dim import dless, Dim
from featurebox.symbol.flow import BaseLoop
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

    select = ['cell volume', 'electron density', 'lattice constants a', 'lattice constants c', 'covalent radii',
              'ionic radii(shannon)',
              'core electron distance(schubert)', 'fusion enthalpy', 'cohesive energy(Brewer)', 'total energy',
              'effective nuclear charge(slater)', 'valence electron number', 'electronegativity(martynov&batsanov)',
              'atomic volume(villars,daams)']  # human select

    select = ['cell volume', 'electron density', ] + [j + "_%i" % i for j in select[2:] for i in range(2)]

    X_frame = data225_import[select]
    y_frame = data225_import['exp_gap']

    X = X_frame.values
    y = y_frame.values

    x, y = shuffle(X, y, random_state=5)
    # data
    data = load_boston()

    c = [6, 3, 4]
    # unit
    from sympy.physics.units import kg

    x_u = [kg] * 13
    y_u = kg
    c_u = [dless, dless, dless]

    x, x_dim = Dim.convert_x(x, x_u, target_units=None, unit_system="SI")
    y, y_dim = Dim.convert_xi(y, y_u)
    c, c_dim = Dim.convert_x(c, c_u)

    z = time.time()

    # symbolset
    pset0 = SymbolSet()
    pset0.add_features(x, y, x_dim=x_dim, y_dim=y_dim, group=None)
    pset0.add_constants(c, dim=c_dim, prob=None)
    pset0.add_operations(power_categories=(2, 3, 0.5),
                         categories=("Add", "Mul", "Sub", "Div", "exp"),
                         self_categories=None)

    # a = time.time()
    bl = BaseLoop(pset=pset0, gen=10, pop=500, hall=1, batch_size=40, re_hall=2,
                  n_jobs=6, mate_prob=0.8, max_value=5,
                  mutate_prob=0.5, tq=True, dim_type=dless,
                  re_Tree=1, store=False, random_state=1,
                  stats={"fitness_dim_max": ["max"], "dim_is_target": ["sum"]},
                  add_coef=True, cal_dim=True, personal_map=False)
    # b = time.time()
    bl.run()
    # c = time.time()
    # print(c - b, b - a, a - z)
