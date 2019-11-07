#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/29 19:47
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
# Just a copy from xenonpy
"""

import os
from itertools import zip_longest
from os import remove

import joblib
import numpy as np
import pandas as pd
from pymatgen import MPRester
from tqdm import tqdm


class Store(object):
    def __init__(self, path=None, filename="filename", prefix: str = None):
        r"""
        store file in path
        :param filename: str,universal filename
        :param path: None, ..
        \data_cluster, or F:data_cluster\data1
        """

        if not prefix:
            prefix = ""
        self._prefix = prefix

        if path is None:
            path = os.getcwd()
        if os.path.exists(path):
            os.chdir(path)
        else:
            os.makedirs(path)
            os.chdir(path)

        self._path = path
        self._filename = ""
        self.default_filename = filename
        self._file_list = []

    def __repr__(self):
        return "store to ({}) with {} file".format(self._path, len(self.stored_file))

    __str__ = __repr__

    def _check_name(self, suffix="csv", file_new_name="filename", model="w"):

        self._filename = file_new_name or self.default_filename

        if os.path.isfile('{}{}.{}'.format(self._prefix, self._filename, suffix)) and model == "w":
            shu1 = 1
            while os.path.isfile('{}{}({}).{}'.format(self._prefix, self._filename, shu1, suffix)):
                shu1 += 1
            self._filename = '{}{}({}).{}'.format(self._prefix, self._filename, shu1, suffix)
        else:
            self._filename = '{}{}.{}'.format(self._prefix, self._filename, suffix)

        if self._filename in self._file_list:
            self._file_list.remove(self._filename)
        self._file_list.append(self._filename)

    def to_csv(self, data, file_new_name=None, model="w"):

        self._check_name("csv", file_new_name, model=model)
        if isinstance(data, (dict, list)):
            data = pd.DataFrame.from_dict(data)
        elif isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        if isinstance(data, pd.DataFrame):
            data.to_csv(path_or_buf="%s" % self._filename, sep=",", na_rep='', float_format=None,
                        columns=None, header=True, index=True, index_label=None,
                        mode=model, encoding=None, )

        else:
            raise TypeError("Not support data_cluster type:%s for csv" % type(data))

    def to_txt(self, data, file_new_name=None, model="w"):
        self._check_name("txt", file_new_name, model=model)
        document = open(self._filename, model)
        document.write(str(data))
        document.close()

    def to_pkl_pd(self, data, file_new_name=None):
        self._check_name("pkl.pd", file_new_name)
        pd.to_pickle(data, self._filename)

    def to_pkl_sk(self, data, file_new_name=None):
        self._check_name("pkl.sk", file_new_name)
        joblib.dump(data, self._filename)

    def remove(self, index_or_name=None):
        """
        remove the indexed file
        :param index_or_name: index or name,default=-1

        """
        if isinstance(index_or_name, str):
            name = index_or_name
            index = -1
        elif isinstance(index_or_name, int):
            name = None
            index = index_or_name
        else:
            name = None
            index = -1
        if not name:
            try:
                files = self._file_list[index]
            except IndexError:
                raise IndexError("No flie or wrong index to remove")
            else:
                if not isinstance(files, list):
                    remove(str(files))
                else:
                    for f in files:
                        remove(str(f))
                del self._file_list[index]

        elif name in self._file_list:
            if not isinstance(name, list):
                remove(str(name))
                self._file_list.remove(name)
            else:
                for f in name:
                    remove(str(f))
                    self._file_list.remove(f)
                self._file_list.remove([])

        else:
            raise NameError("No flie named %s" % index_or_name)

    def withdraw(self):
        """
        delet all stored_file file
        """
        files = self._file_list
        for f in files:
            remove(str(f))
        self._file_list = []
        self._filename = ""

    @property
    def stored_file(self):
        [print(i) for i in self._file_list]
        return self._file_list


def data_fetcher(api_key, mp_ids, elasticity=True):
    """fetch file from pymatgen"""
    print('Will fetch %s inorganic compounds from Materials Project' % len(mp_ids))

    def grouper(iterable, n, fillvalue=None):
        """"
        split requests into fixed number groups
        eg: grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
        Collect data_cluster into fixed-length chunks or blocks"""
        args = [iter(iterable)] * n
        return zip_longest(fillvalue=fillvalue, *args)

    # the following props will be fetched
    mp_props = [
        'band_gap',
        'density',
        'volume',
        'material_id',
        'pretty_formula',
        'elements',
        'efermi',
        'e_above_hull',
        'formation_energy_per_atom',
        'final_energy_per_atom',
        'unit_cell_formula',
        'spacegroup'
        'nelements'
    ]
    if elasticity:
        mp_props.append("elasticity")

    entries = []
    mpid_groups = [g for g in grouper(mp_ids, 10)]

    with MPRester(api_key) as mpr:
        for group in tqdm(mpid_groups):
            mpid_list = [ids for ids in filter(None, group)]
            chunk = mpr.query({"material_id": {"$in": mpid_list}}, mp_props)
            try:
                if elasticity:
                    [entry_i.update(entry_i['elasticity']) for entry_i in chunk if 'elasticity' in entry_i]
            except TypeError:
                pass
            entries.extend(chunk)

    df = pd.DataFrame(entries, index=[e['material_id'] for e in entries])
    # df = df.drop('material_id', axis=1)
    df = df.rename(columns={'unit_cell_formula': 'composition'})
    # df = df['volume_per'] = df['volume']/df['nelements']
    df = df.reindex(columns=sorted(df.columns))

    return df


# def get_ids(api_key="Di2IZMunaeR8vr9w", name_list=None):
#     """
#     support_proprerity = ['energy', 'energy_per_atom', 'volume', 'formation_energy_per_atom', 'nsites',
#     'unit_cell_formula','pretty_formula', 'is_hubbard', 'elements', 'nelements', 'e_above_hull', 'hubbards',
#     'is_compatible', 'spacegroup', 'task_ids',  'band_gap', 'density', 'icsd_id', 'icsd_ids', 'cif',
#     'total_magnetization','material_id', 'oxide_type', 'tags', 'elasticity']
#     """
#     """
#     $gt	>,  $gte >=,  $lt <,  $lte <=,  $ne !=,  $in,  $nin (not in),  $or,  $and,  $not,  $nor ,  $all
#     """
#     m = MPRester(api_key)
#     ids = m.query(criteria={
#         'pretty_formula': {"$in": name_list},
#         # 'nelements': 2,
#         'spacegroup.number': {"$in": [225]},
#         # 'nsites': {"$lt": 5},
#         # 'formation_energy_per_atom': {"$lt": 0},
#         # "elements": {"$in": ["Al", "Co", "Cr", "Cu", "Fe", 'Ni'], "$all": "O"},
#         # "elements": {"$in": list(combinations(["Al", "Co", "Cr", "Cu", "Fe", 'Ni'], 5))}
#     }, properties=["material_id"])
#     print("number %s" % len(ids))
#     return ids

def get_ids(api_key="Di2IZMunaeR8vr9w", name_list=None):
    """
    support_proprerity = ['energy', 'energy_per_atom', 'volume', 'formation_energy_per_atom', 'nsites',
    'unit_cell_formula','pretty_formula', 'is_hubbard', 'elements', 'nelements', 'e_above_hull', 'hubbards',
    'is_compatible', 'spacegroup', 'task_ids',  'band_gap', 'density', 'icsd_id', 'icsd_ids', 'cif',
    'total_magnetization','material_id', 'oxide_type', 'tags', 'elasticity']
    """
    """
    $gt	>,  $gte >=,  $lt <,  $lte <=,  $ne !=,  $in,  $nin (not in),  $or,  $and,  $not,  $nor ,  $all	
    """
    m = MPRester(api_key)
    ids = m.query(criteria={
        # 'pretty_formula': {"$in": name_list},
        'nelements': {"$lt": 3},
        # 'spacegroup.number': {"$in": [225]},
        # 'nsites': {"$lt": 5},
        # 'formation_energy_per_atom': {"$lt": 0},
        # "elements": {"$in": ["Al", "Co", "Cr", "Cu", "Fe", 'Ni'], "$all": "O"},
        # "elements": {"$in": list(combinations(["Al", "Co", "Cr", "Cu", "Fe", 'Ni'], 5))}
    }, properties=["material_id"])
    print("number %s" % len(ids))
    return ids


if __name__ == "__main__":
    list1 = list(
        ['CsCl', 'CsBr', 'CsI', 'CsSb', 'LiF', 'KF', 'RbF', 'CsF', 'MgO', 'CdO', 'MnO', 'VO', 'CaO', 'SrO', 'BaO',
         'EuO', 'ScN', 'YN', 'ErN', 'HoN', 'DyN', 'GdN', 'EuN', 'CeN', 'LiCl', 'TlCl', 'AgCl', 'NaCl', 'RbCl', 'LiBr',
         'TlBr', 'AgBr', 'NaBr', 'KBr', 'RbBr', 'MgSe', 'PbSe', 'CaSe', 'SrSe', 'BaSe', 'YbSe', 'EuSe', 'SmSe', 'PbS',
         'MnS', 'CaS', 'SrS', 'BaS', 'YbS', 'EuS', 'SmS', 'LiI', 'TlI', 'NaI', 'KI', 'RbI', 'YbAs', 'TmAs', 'DyAs',
         'GdAs', 'NdAs', 'SmAs', 'PrAs', 'SmP', 'AsTe', 'GeTe', 'SnTe', 'PbTe', 'CaTe', 'SrTe', 'BaTe', 'YbTe', 'ErTe',
         'GdTe', 'EuTe', 'SmTe', 'LaSb', 'YbSb', 'SmSb', 'PrSb', 'NaF', 'KCl', 'CuBr', 'BeSe', 'ZnSe', 'CdSe', 'HgSe',
         'BeS', 'ZnS', 'CdS', 'AlAs', 'AlP', 'BeTe', 'ZnTe', 'CdTe', 'HgTe', 'AlSb', 'BN', 'SiC3c', 'GaAs', 'InAs',
         'BP', 'GaP', 'InP', 'GaSb', 'InSb', 'CuCl', 'HgS', 'CuI', 'MnTe', 'AgI', 'ZnS', 'ZnSe', 'ZnO', 'AlN', 'GaN',
         'MgTe', 'BeO', 'BN', 'InN', 'SiC', 'MnS'])
    idss = get_ids(api_key="Di2IZMunaeR8vr9w", name_list=list1)
    idss1 = [i['material_id'] for i in idss][:30]
    dff = data_fetcher("Di2IZMunaeR8vr9w", idss1, elasticity=True)
    st = Store(r"C:\Users\Administrator\Desktop")
    st.to_csv(dff, "id_structure")
