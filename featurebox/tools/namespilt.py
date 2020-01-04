# -*- coding: utf-8 -*-

# @Time    : 2019/10/18 14:27
# @Email   : 986798607@qq.ele_ratio
# @Software: PyCharm
# @License: BSD 3-Clause
import copy
import os
import re
from itertools import chain

import pandas as pd

#################
'''产生对应到94元素的表格'''


#################


class Ele:
    def __init__(self, name):
        self.name = [name, ]
        self.num = [1, ]

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            self_new = copy.deepcopy(self)
            self_new.num = [other * _ for _ in self.num]
            return self_new
        else:
            raise TypeError("can not mul {} and  {}".format(self.__str__, other.__str__))

    def __str__(self):
        return "".join(["{}{}".format(i, j) for i, j in zip(self.name, self.num)])

    __repr__ = __str__

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            self_new = copy.deepcopy(self)
            self_new.num = [other * _ for _ in self.num]
            return self_new
        else:
            raise TypeError("can not mul {} and  {}".format(self.__str__, other.__str__))

    def __add__(self, other):
        if isinstance(other, (float, int)):
            raise TypeError("can not add {} and number {}".format(self.__str__, other))
            pass
        elif isinstance(other, Ele):
            self_new = copy.deepcopy(self)
            for i, j in zip(other.name, other.num):
                if i in self.name:
                    index = self.name.index(i)
                    self_new.num[index] += j
                else:
                    self_new.name.append(i)
                    self_new.num.append(j)
            return self_new

    def __pos__(self):
        return copy.deepcopy(self)

    @property
    def to_dict(self):
        return {i: j for i, j in zip(self.name, self.num)}

    @property
    def to_item(self):
        # print(list(zip(self.x_name, self.num)))
        return list(chain.from_iterable(zip(self.name, self.num)))


name_ele = ['H', 'He', 'Li', 'Be', 'B', 'C', 'numeric', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
            'K',
            'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
            'Y',
            'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
            'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re',
            'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np',
            'Pu']

for ele_i in name_ele:
    locals()[ele_i] = Ele(ele_i)


def _init_dict(element_n=None):
    if not element_n:
        element_n = len(name_ele)
    dict0 = dict()
    name_list_n = ["Abandon"] + name_ele
    for i in name_list_n[:element_n]:
        dict0[i] = 0
    return dict0


def fea_dict(ele, element_n=None):
    dict0 = _init_dict(element_n=element_n)
    dicti = copy.copy(dict0)
    num = ele.num
    sname = ele.name
    for j, k in zip(sname, num):
        if j in dicti:
            dicti[j] = k
        else:
            dicti["Abandon"] = 1
    return dicti


def transform(names):
    c = []
    for i in names:
        try:
            com = eval(i)
        except NameError as e:
            print("{}, from {}".format(e, i))
            com = Ele(i)
        c.append(com)
    folds = pd.DataFrame([_i.to_item for _i in c], [str(_) for _ in c])
    expands = pd.DataFrame([fea_dict(_i) for _i in c], [str(_) for _ in c])
    folds.to_csv(r'folds.csv')
    expands.to_csv(r'expands.csv')


def load_csv_name(path=r'C:\Users\ww\Desktop', dataname=r'jqxx.txt', skiprow=2):
    try:
        os.chdir(path)
        open(dataname)
    except IOError:
        raise IOError('No file:{} in {} '.format(dataname, path))
    else:
        data = pd.read_csv(dataname)
        feature_name = data.columns
        second_line = list(data.loc[0])
        third_line = list(data.loc[1])
        first_col = list(data.iloc[skiprow:, 0])
        return feature_name, second_line, third_line, first_col


def _pre_add(namei):
    a_ = ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar', 'Ca', 'Sc', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
          'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
          'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
          'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At',
          'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Np', 'Pu']
    b = ['H', 'B', 'C', 'numeric', 'O', 'F', 'K', 'P', 'S', 'V', 'Y', 'I', 'W', 'U']

    num = re.findall(r"\d+\.?\d*", namei)

    namei_new = ""
    namei3 = namei
    j = 0
    while j < len(num):
        s1, s2, namei3 = namei3.partition(num[j])
        namei_new = r"".join([namei_new, s1, "_", s2])
        j += 1
    if namei3:
        namei_new += namei3

    namej = namei_new

    for x in a_:
        namej = re.sub(x, x + '1', namej)
    for x in b:
        namej = re.sub(x, x + '1', namej)
    needsub = re.findall(r'[A-Z]1[][a-z]', namej)
    for i in needsub:
        namej = re.sub(i, i[0] + i[2], namej)
    namej = re.sub(r'1?_', "", namej)

    fin = re.findall(r'\)[A-Z][a-z]*', namej)
    for i in fin:
        namej = re.sub(re.escape(i), ")1" + i[1:], namej, 1)

    return namej


def _bracket_follow(s):
    """
    for situation that the number of () before the ()
    if add this function  make sure the first number below ")" have "+" !!!
    Parameters
    ----------
    s

    Returns
    -------

    """
    pre = re.findall(r'\d+\.?\d*\(.+\)\D*', s)
    long = len(pre)
    while long > 0:
        i = pre[0]
        a_, b, c = i.partition("(")
        d, e, f = c.partition(")")
        res = r"(" + d + r")" + a_ + f
        # s = re.sub(i, res, s, 1)
        s = s.replace(i, res, 1)
        pre = re.findall(r'\d+\.?\d*\(.\)\D*', s)
        long = len(pre)
    return s


def _add_mul(s):
    ret2 = re.findall(r'\d+\.?\d*', s)
    s_new = ""
    s3 = s
    j = 0
    while j < len(ret2):
        s1, s2, s3 = s3.partition(ret2[j])
        s_new = r"".join([s_new, "+", s1, "*", s2])
        j += 1
    if s3:
        s_new += "+" + s3
    s_new = s_new.replace(r"+ +", r"+", 1)
    s_new = s_new.replace(r"+)", r")")
    return s_new


def substutude(s):
    # s = _bracket_follow(s)
    s = _pre_add(s)
    s = _add_mul(s)
    return s


if __name__ == '__main__':
    os.chdir(r'C:\Users\Administrator\Desktop')
    a = [substutude(i) for i in [
        # r"0.8(TiLa2)H2", r"(Ti1.24La)0.2H2", r"(Ti1.24La)0.2", '(Ti1.24La)',
        '(BaZr0.2Ti0.8O3)0.9(Ba0.7Ca0.3TiO3)0.1',
        '(Ba0.95Ca0.05)+(Zr0.1Ti0.9)O3',
        '(Ba(Zr0.2Ti0.8)O3)0.5 + ((Ba0.7Ca0.3)TiO3)0.5',
        'Ba0.95Sr0.05+(Ti0.95Zr0.05)0.975Sn0.025O3',
        "Ba0.95Sr0.05(Ti0.95Zr0.05)O3",
        "Ba0.9Sr0.1(Ti0.9Zr0.1)0.95Sn0.05O3",
        '(Ti1.24La3)2',
        "((Ti1.24)2P2)1H0.2", "((Ti1.24)2)1H0.2", "((Ti1.24))1H0.2",
        r"(TiLa)H2", r"(TiLa)0.2H2",
        r"(TiLa)0.2", '(TiLa)',
        '(TiLa3)', '(TiLa)', '(TiLa3)2',
        "((Ti)2P2)1H0.2", "((Ti)2)1H0.2", "((Ti))1H0.2"
    ]]
    transform(a)
