# -*- coding: utf-8 -*-

# @Time    : 2020/1/8 16:28
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import numpy as np


def get_rol(axis_all, axis):
    return [j for i, j in enumerate(axis_all) if i != axis]


def data_resize(x):
    return x


def rotatespilt(TDarray, angle=10, axis=0, axis_site=None, resize=False):
    """
    Rotate the np.array with an center and axis.

    Parameters
    ----------
    TDarray:np.ndarray
        three Dimension np.ndarray
    angle:int
        angle from 0 to 180
    axis:int
        the rotate axis
    axis_site:tuple
        rotate center
    resize:bool
        # todo
    Returns
    -------
    result:list of np.ndarray
        All 2d array with different angle.

    """
    assert TDarray.ndim == 3
    angle = np.pi / 180 * angle

    slices = []
    for j, i in enumerate(np.arange(0, np.pi, angle)):
        try:
            data_all = _rotatespilt(TDarray, angle=i, axis=axis, axis_site=axis_site)
            if resize:
                data_all = data_resize(data_all)
            slices.append(data_all)
        except UserWarning:
            print('except the angle {}'.format(i / np.pi * 180))
    return slices


def _rotatespilt(TDarray, angle=np.pi / 4, axis=0, axis_site=None):
    # 得到两轴长度
    if axis_site is None:
        axis_site = (0, 0)
    shape_rol = get_rol(TDarray.shape, axis)
    add_site = [int(i * j) for i, j in zip(axis_site, shape_rol)]
    add_site = np.array(add_site).reshape((-1, 1))
    if angle == np.pi / 2:
        if axis == 0:
            data = np.squeeze(TDarray[:, add_site[0], :])
        elif axis == 1:
            data = np.squeeze(TDarray[add_site[0], :, :])
        else:
            data = np.squeeze(TDarray[:, :, add_site[0]])
        data_all = data
    else:
        # 得到两轴
        # axis_all = list(range(TDarray.ndim))
        # axis_rol = get_rol(axis_all, axis)

        # 网格
        meshes = np.meshgrid(*[range(i) for i in shape_rol])
        meshes = np.array([_.ravel() for _ in meshes])

        # 距离
        s1 = meshes[0] - add_site[0]
        s2 = np.tan(angle) * (meshes[1] - add_site[1])
        distance = np.abs(s1 - s2)
        # show = np.array((s1, meshes[1] - add_site[1], s2, distance))
        # 间距
        site = np.where(distance < 2)[0]
        point = meshes[:, site]

        point_max = np.max(point, axis=1)
        point_min = np.min(point, axis=1)
        point_dis = point_max - point_min
        range_ = int(np.sqrt(np.sum((point_dis ** 2)))) + 1

        if angle < 1.57:
            start_point = point_min
            # end_point = point_max
        else:
            start_point = (point_min[0], point_max[1])
            # end_point = (point_max[0], point_min[1])

        dis = point - np.array(start_point).reshape(-1, 1)
        dis = np.sqrt(np.sum((dis ** 2), axis=0))
        point_near = []
        for i in range(range_):
            tem = np.abs(dis - i)
            if len(tem) <= 3:
                raise UserWarning(
                    "the slice is at a corner,with little point,please change the axis_site or the angle ")
            else:
                near = np.argpartition(tem, 3)[:3]
            point_near.append(point[:, near])

        data_all = []
        for point_near_i in point_near:
            if axis == 0:
                data = [TDarray[:, i, j] for i, j in point_near_i.T]
            elif axis == 1:
                data = [TDarray[i, :, j] for i, j in point_near_i.T]
            else:
                data = [TDarray[i, j, :] for i, j in point_near_i.T]

            data = np.mean(np.array(data), axis=0)
            data_all.append(data)
        data_all = np.array(data_all).T

    return data_all


# a = np.array(
#     [[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], ],
#      [[3, 3, 2, 3, 3], [3, 3, 2, 3, 3], [3, 3, 2, 3, 3]]])
a = np.zeros((11, 12, 13))
s = rotatespilt(a, angle=10, axis=0, axis_site=(0.5, 0.5))
