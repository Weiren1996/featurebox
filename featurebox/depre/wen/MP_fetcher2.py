from itertools import zip_longest
from pymatgen import MPRester
from tqdm import tqdm
import os
import warnings
from os import remove
import numpy as np
import pandas as pd
from sklearn.externals import joblib

from featurebox.tools.exports import Store

warnings.filterwarnings("ignore")

def data_fetcher(api_key, mp_ids, elasticity=True):
    #     print('Will fetch %s inorganic compounds from Materials Project' % len(mp_ids))

    # split requests into fixed number groups
    # eg: grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    def grouper(iterable, n, fillvalue=None):
        """Collect data_cluster into fixed-length chunks or blocks"""
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
        'spacegroup',
        'nelements'
    ]
    if elasticity:
        mp_props.append("elasticity")

    entries = []
    mpid_groups = [g for g in grouper(mp_ids, 50)]

    with MPRester(api_key) as mpr:
        for group in tqdm(mpid_groups):
            mpid_list = [ids for ids in filter(None, group)]
            chunk = mpr.query({"material_id": {"$in": mpid_list}}, mp_props)

            if elasticity:
                for entry_i in chunk:
                    if 'elasticity' in entry_i and entry_i['elasticity'] is not None:
                        entry_i.update(entry_i['elasticity'])
                # [entry_i.update(entry_i['elasticity']) for entry_i in chunk if 'elasticity' in entry_i]

            entries.extend(chunk)

    df = pd.DataFrame(entries, index=[e['material_id'] for e in entries])
    # df = df.drop('material_id', axis=1)
    df = df.rename(columns={'unit_cell_formula': 'composition'})
    # df = df['volume_per'] = df['volume']/df['nelements']
    df = df.reindex(columns=sorted(df.columns))

    return df


def get_ids(api_key="Di2IZMunaeR8vr9w", name_list=[]):
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
                            # 'nelements': 2,
                            'spacegroup.number': {"$in": [225]},
                            # 'nsites': {"$lt": 5},
                            # 'formation_energy_per_atom': {"$lt": 0},
                            # "elements": {"$in": ["Al", "Co", "Cr", "Cu", "Fe", 'Ni'], "$all": "O"},
                            # "elements": {"$in": list(combinations(["Al", "Co", "Cr", "Cu", "Fe", 'Ni'], 5))}
                            }, properties=["material_id"])
    print("number %s" % len(ids))
    return ids


if __name__ == "__main__":

    idss = get_ids(api_key="Di2IZMunaeR8vr9w")
    idss1 = [i['material_id'] for i in idss]
    dff = data_fetcher("Di2IZMunaeR8vr9w", idss1, elasticity=True)
    st = Store(r"C:\Users\Administrator\Desktop")
    st.to_csv(dff, "id_structure")
