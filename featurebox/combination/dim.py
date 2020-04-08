# -*- coding: utf-8 -*-

# @Time    : 2019/11/12 15:35
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
# -*- coding: utf-8 -*-

import numbers

import numpy as np
import numpy.core.numeric as numeric
import sympy


def dim_func():
    """dim function """

    def my_abs(dim):

        return dim

    def my_sqrt(dim):

        return dim.__pow__(0.5)

    def my_exp(dim):

        if isinstance(dim, Dim):
            if dim == dless:
                return dless
            else:
                return dnan

    my_log = my_cos = my_sin = my_exp

    my_funcs = {"Abs": my_abs, "exp": my_exp, "log": my_log, 'cos': my_cos, 'sin': my_sin, 'sqrt': my_sqrt}
    return my_funcs


class Dim(numeric.ndarray):
    """re define the Dimension of sympy """

    def __new__(cls, data, dtype=np.float16, copy=True):

        assert isinstance(data, (numeric.ndarray, list))

        arr = numeric.array(data, dtype=dtype, copy=copy)
        try:
            arr.reshape((7, 1))
        except BaseException:
            raise IndexError("the size of datamnist is not 7")

        shape = arr.shape

        ret = numeric.ndarray.__new__(cls, shape, arr.dtype,
                                      buffer=arr,
                                      order='F')
        return ret

    def __init__(self, data):
        _ = data
        self.unit = ["kg", "m", "s", "A", "K", "mol", "cd"]

    def __add__(self, other):

        if isinstance(other, Dim) and self != other:
            if other == dless:
                return self
            elif self == dless:
                return other
            else:
                return dnan
        elif isinstance(other, Dim) and self == other:
            return self

        elif isinstance(other, (numbers.Real, sympy.Rational, sympy.Float)):
            return self
        else:
            return dnan

    def __sub__(self, other):
        return self + other

    def __pow__(self, other):
        return self._eval_power(other)

    def _eval_power(self, other):
        if isinstance(other, (numbers.Real, sympy.Rational, sympy.Float)):
            return Dim(np.array(self) * other)
        else:
            return dnan

    def __mul__(self, other):
        if isinstance(other, Dim):
            return Dim(np.array(self) + np.array(other))
        elif isinstance(other, (numbers.Real, sympy.Rational, sympy.Float)):
            return self
        else:
            return dnan

    def __div__(self, other):

        if isinstance(other, Dim):
            return Dim(np.array(self) - np.array(other))
        elif isinstance(other, (numbers.Real, sympy.Rational, sympy.Float)):
            return self
        else:
            return dnan

    def __rdiv__(self, other):
        # return other*spath._eval_power(-1)
        if isinstance(other, (numbers.Real, sympy.Rational, sympy.Float)):
            return self.__pow__(-1)
        else:
            return dnan

    def __abs__(self):
        return self

    def __rpow__(self, other):
        return dnan

    def __eq__(self, other):
        return all(np.equal(self, other))

    def __ne__(self, other):
        return not all(np.equal(self, other))

    def __neg__(self):
        return self

    def __pos__(self):
        return self

    @property
    def allisnan(self):
        return all(np.isnan(self))

    @property
    def anyisnan(self):
        return any(np.isnan(self))

    @property
    def isfloat(self):
        return any(np.modf(self)[0])

    @property
    def isinteger(self):
        return not any(np.modf(self)[0])

    def is_same_base(self, others):
        if isinstance(others, Dim):
            npself = np.array(self)
            npothers = np.array(others)
            x1 = np.linalg.norm(npself)
            x2 = np.linalg.norm(npothers)

            if others ** x1 == self ** x2:
                return True
            else:
                return False
        else:
            return False

    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__

    def __str__(self):
        strr = "".join(["{}^{}*".format(i, j) for i, j in zip(self.unit, self)])[:-1]

        return strr


dnan = Dim(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]))
dless = Dim(np.array([0, 0, 0, 0, 0, 0, 0]))

# if __name__=="__main__":
#
#
# ta = time.time()
#
# x = Dim([1, 2, 3, 4, 5, 6, 7])
# a = Dim(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, ]))
# b = Dim(np.array([1, 1, 1, 1, 1, 1, 1]))

# c = b + 8
# d = 8 + b
# e = b + a
# f = a + b
# g = b + x
#
# g1 = a + dless
# g2 = dless+a
# g3 = dless + x
# g4 = x+dless
# g5 = dless + 1
# g6 = 1 + dless

# c = b - 8
# d = 8 - b
# e = b - a
# f = a - b
# g = b - x
#
# c = b * 8
# d = 8 * b
# e = b * a
# f = a * b
# g = b * x
#
# c = b / 8
# d = 8 / b
# e = b / a
# f = a / b
# g = b / x
#
# c = b ** 8
# d = 8 ** b
# e = b ** a
# f = a ** b
# g = b ** x

# h = abs(b)
# j = abs(a)
#
# h = -b
# j = -a

# xx = np.copy(a)
# print(a == dnan)
# print(a is a)
# print(a != b)
# print(a == 1)
# #
# k = dim_func()["exp"](a)
# l = dim_func()["exp"](b)
# m = dim_func()["exp"](dless)
#
#
# tb = time.time()
# print(tb - ta)
