# -*- coding: utf-8 -*-

# @Time    : 2019/12/30 14:19
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import json

# import requests
# from pymatgen import MPRester
#
# data = {
#     'criteria': {
#         'elements': {'$in': ['Li', 'Na', 'K'], '$all': ['O']},
#         'nelements': 2,
#     },
#     'properties': [
#         'formula',
#         'formation_energy_per_atom',
#     ]
# }
# r = requests.post('https://materialsproject.org/rest/v2/query',
#                  headers={'X-API-KEY': "Di2IZMunaeR8vr9w"},
#                  data={k: json.dumps(v) for k,v in data.items()})
# response_content = r.json() # a dict
#
# m = MPRester("Di2IZMunaeR8vr9w")
# a=m.get_data( "mp-1234", data_type='vasp')
e = open(r"C:\Users\Administrator\Desktop\a.txt")
a = json.load(e, encoding="UTF-8")
