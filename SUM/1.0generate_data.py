# -*- coding: utf-8 -*-

# @Time   : 2019/6/13 18:47
# @Author : Administrator
# @Project : feature_toolbox
# @FileName: 3.0select_method.py
# @Software: PyCharm

import pandas as pd
from pymatgen import Composition
from sklearn.svm import LinearSVR

from featurebox.featurizers.compositionfeaturizer import DepartElementFeaturizer
from featurebox.tools.exports import Store

LinearSVR
"""
this is a description
"""
if __name__ == "__main__":

    store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp')

    com_data = pd.read_excel(r'C:\Users\Administrator\Desktop\band_gap_exp\init_band_data.xlsx',
                             sheet_name='binary_4_structure')

    """for element site"""
    element_table = pd.read_excel(r'F:\machine learning\feature_toolbox1.0\featurebox\data\element_table.xlsx',
                                  header=4, skiprows=0, index_col=0)
    """get x_name and abbr"""


    def get_abbr():
        abbr = list(element_table.loc["abbrTex"])
        name = list(element_table.columns)
        name.extend(['face_dist1', 'vor_area1', 'face_dist2', 'vor_area2', "destiny", 'volume', "ele_ratio"])
        abbr.extend(['$d_{vf1}$', '$S_{vf1}$', '$d_{vf2}$', '$S_{vf2}$', r"$\rho_c$", "$V_c$", "$ele_ratio$"])
        return name, abbr


    name_and_abbr = get_abbr()
    store.to_pkl_pd(name_and_abbr, "name_and_abbr")

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
        'electronegativity(martynov&batsanov)',
        'electronegativity(pauling)',
        'electronegativity(alfred-rochow)',

        'volume atomic(villars,daams)',

    ]

    select_element_table = element_table[feature_select]

    """transform composition to pymatgen Composition"""
    composition = pd.Series(map(eval, com_data['composition']))
    composition_mp = pd.Series(map(Composition, composition))

    """get ele_ratio"""


    def comdict_to_df(composition_mp):
        composition_mp = pd.Series([i.to_reduced_dict for i in composition_mp])
        com = [[j[i] for i in j] for j in composition_mp]
        com = pd.DataFrame(com)
        colu_name = {}
        for i in range(com.shape[1]):
            colu_name[i] = "com_%s" % i
        com.rename(columns=colu_name, inplace=True)
        return com


    ele_ratio = comdict_to_df(composition_mp)

    """get structure"""
    # with MPRester('Di2IZMunaeR8vr9w') as m:
    #     ids = [i for i in com_data['material_id']]
    #     structures = [m.get_structure_by_material_id(i) for i in ids]
    # store.to_pkl_pd(structures, "id_structures")
    # id_structures = pd.read_pickle(
    #     r'C:\Users\Administrator\Desktop\band_gap_exp\1.generate_data\id_structures.pkl.pd')

    """get departed element feature"""
    departElementProPFeature = DepartElementFeaturizer(elem_data=select_element_table, n_composition=2, n_jobs=4, )
    departElement = departElementProPFeature.fit_transform(composition_mp)
    """join"""
    depart_elements_table = departElement.set_axis(com_data.index.values, axis='index', inplace=False)
    ele_ratio = ele_ratio.set_axis(com_data.index.values, axis='index', inplace=False)

    all_import_title = com_data.join(ele_ratio)
    all_import_title = all_import_title.join(depart_elements_table)

    store.to_csv(all_import_title, "all_import_title")

    all_import = all_import_title.drop(
        ['name_number', 'name_number', "name", "structure", "structure_type", "space_group", "reference", 'material_id',
         'composition', "com_0", "com_1"], axis=1)

    store.to_csv(all_import, "all_import")
