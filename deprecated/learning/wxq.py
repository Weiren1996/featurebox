#!/usr/bin/python
# coding:utf-8

"""
@author: wangchangxin
@contact: 986798607@qq.com
@software: PyCharm
@file: wxq.py
@time: 2020/3/17 20:26
"""
import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def move(datas, x_, x, y_, y):
    for m, j in product(np.arange(x_, x, 0.01), np.arange(y_, y, 0.01)):
        print(m, j)
        yield np.column_stack((datas[:, 1] + j, datas[:, 0] + m))


def distance(req, sha1):
    sha = np.copy(sha1)
    poi_sha = []
    poi_req = []
    for l, k in enumerate(req):
        sh = sha - k
        dis = np.sqrt(np.sum(sh ** 2, axis=1))
        if np.min(dis) < 0.1:
            index = np.argmin(dis)
            poi_sha.append(index)
            poi_req.append(l)
            # sha = np.delete(sha, index, 0)
            sha[index] = 10000
    return poi_sha, poi_req


def change_cwd(file):
    driver, name = os.path.split(file)
    os.chdir(driver)


if __name__ == "__main__":

    file = r"C:\Users\Administrator\Desktop\1.xlsx"
    change_cwd(file)

    data = pd.read_excel(file)
    required = data[["required_X", "required_y"]].values
    shape = data[["shape_X", "shape_y"]].values
    required = required[:27]
    #
    #
    # #
    points = 0
    count = 0
    point_req = 0
    req_i = 0
    for i in move(required, -1, 6, 0.2, 1.0):

        po_sha, po_re = distance(i, shape)
        cou = len(po_sha)
        if cou > count:
            points = po_sha
            count = cou
            point_req = po_re
            req_i = i
    find_shape = shape[points, :]
    find_req = req_i[point_req, :]
    find_data = np.concatenate((find_shape, find_req), axis=1)
    #
    print("结果在shape中的位置为", points, "(从0开始计数)")
    print("共计%d个" % count)
    pd.DataFrame(find_data).to_csv("finsd_shape.xlsx")

    print("保存至 %s" % os.getcwd())
    plt.scatter(shape[:, 0], shape[:, 1], marker="^")
    plt.scatter(find_shape[:, 0], find_shape[:, 1], c="r", marker="^")
    plt.scatter(req_i[:, 0], req_i[:, 1], c="b", marker="o")
    plt.scatter(find_req[:, 0], find_req[:, 1], c="g", marker="o")
    plt.show()
