#!/usr/bin/python
#coding:utf-8

"""
@author: wangchangxin
@contact: 986798607@qq.com
@software: PyCharm
@file: ttt.py
@time: 2020/5/14 23:13
"""
from functools import partial

import numpy as np
def fu(a,b):
    print([i+b for i in a])

s = [1,2,3,4]
pfunc = partial(fu, a=s)
pfunc(b=1)