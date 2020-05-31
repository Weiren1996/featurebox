# -*- coding: utf-8 -*-

# @Time    : 2020/1/6 9:41
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from pymatgen.io.vasp import inputs

from featurebox.tools.imports import BatchFile

# Kpoints
# Poscar
# Incar
#
# VaspInput
# Chgcar
#
# Elfcar
# Oszicar
# Outcar

if __name__ == "__main__":
    a = BatchFile(r"C:\Users\Administrator\Desktop\file")
    # a.filter_dir_name("line",layer=-1)
    a.filter_dir_name(exclud="band_line")
    a.filter_file_name("POSCAR")
    lists = a.merge()
    # a.to_path(r"C:\Users\Administrator\Desktop\newss", add_dir=[-3, -2, -1], flatten=False)

    # ii=inputs.Incar
    # ii=inputs.Kpoints
    ii = inputs.Poscar

    for i, j in enumerate(lists):
        isi = ii.from_file(lists[0])
        # print(ii)
