# -*- coding: utf-8 -*-

# @Time   : 2019/6/13 18:47
# @Author : Administrator
# @Project : feature_toolbox
# @FileName: 3.0select_method.py
# @Software: PyCharm

import pandas as pd
from pymatgen import Composition
from featurebox.featurizer.elementfeature import DepartElementProPFeaturizer
from featurebox.tools.exports import Store

"""
this is a description
"""
if __name__ == "__main__":
    store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp_last\1.generate_data')
    com_data = pd.read_excel(r'C:\Users\Administrator\Desktop\band_gap_exp_last\init_band_data.xlsx',
                             sheet_name='binary_4_structure')
    composition = pd.Series(map(eval, com_data['composition']))
    composition_mp = pd.Series(map(Composition, composition))
    composition_mp = pd.Series([i.to_reduced_dict for i in composition_mp])

    com = [[j[i] for i in j] for j in composition_mp]
    com = pd.DataFrame(com)
    colu_name = {}
    for i in range(com.shape[1]):
        colu_name[i] = "com_%s" % i
    com.rename(columns=colu_name, inplace=True)

    # with MPRester('Di2IZMunaeR8vr9w') as m:
    #     ids = [i for i in com_data['material_id']]
    #     structures = [m.get_structure_by_material_id(i) for i in ids]
    # store.to_pkl_pd(structures, "id_structures")

    id_structures = pd.read_pickle(
        r'C:\Users\Administrator\Desktop\band_gap_exp_last\1.generate_data\id_structures.pkl.pd')

    """for element site"""
    element_table = pd.read_excel(r'F:\machine learning\feature_toolbox1.0\featurebox\data\element_table.xlsx',
                                  header=4, skiprows=0, index_col=0)
    element_table = element_table.iloc[5:, 7:]
    feature_select = [
        'lattice constants a',
        'lattice constants b',
        'lattice constants c',
        'radii atomic(empirical)',
        'radii atomic(clementi)',
        'radii ionic(pauling)',
        'radii ionic(shannon)',
        'radii covalent',
        'radii covalent 2',
        'radii metal(waber)',
        'distance valence electron(schubert)',
        'distance core electron(schubert)',
        'radii pseudo-potential(zunger)',

        'energy ionization first',
        'energy ionization second',
        'energy ionization third',
        'enthalpy atomization',
        'enthalpy vaporization',
        'latent heat of fusion',
        'latent heat of fusion 2',
        'energy cohesive brewer',
        'total energy',

        'electron number',
        'valence electron number',
        'charge nuclear effective(slater)',
        'charge nuclear effective(clementi)',
        'periodic number',
        'group number',
        'electronegativity(martynov&batsanov)',
        'electronegativity(pauling)',
        'electronegativity(alfred-rochow)',

        'volume atomic(villars,daams)',

    ]
    select_element_table = element_table[feature_select]
    """"""
    departElementProPFeature = DepartElementProPFeaturizer(elements=select_element_table, n_composition=2, n_jobs=4,
                                                           )
    departElement = departElementProPFeature.fit_transform(composition_mp)
    depart_elements_table = departElement.set_axis(com_data.index.values, axis='index', inplace=False)
    com = com.set_axis(com_data.index.values, axis='index', inplace=False)
    all_import = com_data.join(com)
    all_import = all_import.join(depart_elements_table)

    store.to_csv(all_import, "all_import")
