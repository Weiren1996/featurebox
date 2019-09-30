#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/8/12 17:44
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
this is a description
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from featurebox.selection.corr import Corr
from featurebox.tools import show
from featurebox.tools.exports import Store
from featurebox.tools.imports import Call
from featurebox.tools.show import cof_sel_plot, cof_plot
from featurebox.tools.tool import index_to_name

store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp_last\2.correction_analysis')
data = Call(r'C:\Users\Administrator\Desktop\band_gap_exp_last\1.generate_data')
all_import_structure = data.csv.all_import_structure
data_import = all_import_structure.drop(
    ['name_number', 'name_number', "name", "structure", "structure_type", "space_group", "reference", 'material_id',
     'composition', "com_0", "com_1", 'face_dist0', 'vor_area0'], axis=1)
data225_import = data_import.iloc[np.where(data_import['group_number'] == 225)[0]]

X_frame = data225_import.drop(['exp_gap', 'group_number'], axis=1)
y_frame = data225_import['exp_gap']
X = X_frame.values
y = y_frame.values


def get_abbr(X_frame_name):
    element_table = pd.read_excel(r'F:\machine learning\feature_toolbox1.0\featurebox\data\element_table.xlsx',
                                  skiprows=0, index_col=0)
    name = list(element_table.loc["name"])
    abbr = list(element_table.loc["abbrTex"])
    name.extend(['face_dist1', 'vor_area1', 'face_dist2', 'vor_area2', "destiny", 'volume', "com"])
    abbr.extend(['$d_{vf1}$', '$S_{vf1}$', '$d_{vf2}$', '$S_{vf2}$', r"$\rho_c$", "$V_c$", "$com$"])
    index = [name.index(i) for i in X_frame_name]
    abbr = np.array(abbr)[index]
    return abbr

corr = Corr(threshold=0.90, muti_grade=2, muti_index=[4, len(X)])
corr.fit(X_frame)
cof_list = corr.count_cof()


# plot

X_frame_name = corr.transform(X_frame.columns.values)
X_frame_name = [i.replace("_0", "") for i in X_frame_name]
X_frame_abbr = get_abbr(X_frame_name)
cov = pd.DataFrame(corr.cov_shrink)
cov = cov.set_axis(X_frame_abbr, axis='index', inplace=False)
cov = cov.set_axis(X_frame_abbr, axis='columns', inplace=False)

import seaborn as sns

fig = plt.figure()
fig.add_subplot(111)
sns.heatmap(cov, vmin=-1, vmax=1, cmap="bwr", linewidths=0.3, xticklabels=True, yticklabels=True, square=True,
            annot=True, annot_kws={'size': 3})
plt.show()

list_name = [index_to_name(i, X_frame_name) for i in corr.list_count]
list_abbr = [index_to_name(i, X_frame_abbr) for i in corr.list_count]

# 2
select = ['volume', 'destiny', 'lattice constants a', 'lattice constants c', 'radii covalent', 'radii ionic(shannon)',
          'distance core electron(schubert)', 'latent heat of fusion', 'energy cohesive brewer', 'total energy',
          'charge nuclear effective(slater)', 'valence electron number', 'electronegativity(martynov&batsanov)',
          'volume atomic(villars,daams)']

index = [X_frame_name.index(i) for i in select]
abbr = np.array(X_frame_abbr)[index]
cov_select = corr.cov_shrink[index][:, index]

store.to_csv(cov_select, "cov_select")
store.to_csv(list_name, "list_name")
store.to_txt(list_abbr, "list_abbr")

store.to_csv(cov, "cov")
store.to_txt(select, "list_name_select")
store.to_txt(abbr, "list_abbr_select")

cof_sel_plot(cov_select, abbr, threshold=0.70)
cof_plot(cov_select, abbr)
