# -*- coding: utf-8 -*-

import numpy as np
# @Time    : 2019/11/16 15:58
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import pandas as pd

a = pd.read_pickle(r'C:\Users\Administrator\Desktop\band_gap_exp\3.sum\filename.pkl.pd')
s = a["gen1"]

# t= [list(i.values()) for i in s.values()]
# tt=np.array(t)

lists = []
for si, s in enumerate(a.values()):
    t = [list(i.values()) for i in s.values()]
    tt = np.array(t)
    num = np.full_like(tt, si, )
    tt = np.concatenate((tt, num), axis=1)
    lists.append(tt)
result = np.concatenate(lists, axis=0)
