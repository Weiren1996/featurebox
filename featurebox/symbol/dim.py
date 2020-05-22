#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Time    : 2019/11/12 15:13
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

"""
Notes:
    this some of this part are a copy from deap
"""


from __future__ import division

import numbers
import numpy as np
import numpy.core.numeric as numeric
import sympy
from numpy.linalg import matrix_rank
from sklearn.utils import check_array
from sympy import Add, Mul, Pow, Tuple, sympify
from sympy.core.compatibility import reduce, Iterable
from sympy.physics.units import Dimension
from sympy.physics.units.quantities import Quantity
from sympy.physics.units.systems import SI


def dim_map():
    """expr to dim function """

    def my_abs(dim):
        if isinstance(dim, (numbers.Real, sympy.Rational, sympy.Float)):
            return dless
        else:
            return dim

    def my_sqrt(dim):

        return dim.__pow__(0.5)

    my_self = my_flat = my_abs

    def my_exp(dim):

        if isinstance(dim, Dim):
            if dim == dless:
                return dless
            else:
                return dnan
        elif isinstance(dim, (numbers.Real, sympy.Rational, sympy.Float)):
            return dless
        else:
            return dless

    my_log = my_cos = my_sin = my_exp

    my_funcs = {"Abs": my_abs, "exp": my_exp, "log": my_log, 'cos': my_cos, 'sin': my_sin,
                'sqrt': my_sqrt, "Flat": my_flat, "Self": my_self}
    return my_funcs


def check_dimension(x, y=None):
    """
    check the consistency of dimension.
    Parameters
    ----------
    x: list of Dim
    y:Dim

    Returns
    -------
    bool
    """
    if y is not None:
        x.append(y)
    x = np.array(x).T
    x = check_array(x, ensure_2d=True)
    x = x.astype(np.float64)
    det = matrix_rank(x)
    che = []
    for i in range(x.shape[1]):
        x_new = np.delete(x, i, 1)
        det2 = matrix_rank(x_new)
        che.append(det - det2)
    sum(che)

    if sum(che) == 0:
        return True
    else:
        return False


class Dim(numeric.ndarray):
    """Redefine the Dimension of sympy, the default dimension SI system with 7 number,
    1.can be constructed by list of number.
    1.can be translated from a sympy.physics.unit.
        >>>from sympy.physics.units import N
        >>>scale,dim = Dim.convert_to_Dim(N)
        #inverse back
        >>>Dim.inverse_convert(dim, scale=scale, target_units=None, unit_system="SI")
    """
    __slots__ = ("unit", "unit_map", "dim")

    def __new__(cls, data, dtype=np.float16, copy=True):

        assert isinstance(data, (numeric.ndarray, list))

        arr = numeric.array(data, dtype=dtype, copy=copy)

        arr.reshape((1, -1))
        if arr.shape[0] > 7:
            raise UserWarning("The number of dim more than 7 SI")
        shape = arr.shape

        ret = numeric.ndarray.__new__(cls, shape, arr.dtype,
                                      buffer=arr,
                                      order='F')

        return ret

    def __init__(self, data):
        _ = data
        self.unit_map = {'meter': "m", 'kilogram': "kg", 'second': "s",
                         'ampere': "A", 'mole': "mol", 'candela': "cd", 'kelvin': "K"}
        self.unit = SI._base_units
        self.dim = ['length', 'mass', 'time', 'current', 'amount_of_substance',
                    'luminous_intensity', 'temperature']

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

    @staticmethod
    def _get_conversion_matrix_for_expr(expr, target_units, unit_system):
        from sympy import Matrix

        dimension_system = unit_system.get_dimension_system()

        expr_dim = Dimension(unit_system.get_dimensional_expr(expr))
        dim_dependencies = dimension_system.get_dimensional_dependencies(expr_dim, mark_dimensionless=True)
        target_dims = [Dimension(unit_system.get_dimensional_expr(x)) for x in target_units]
        canon_dim_units = [i for x in target_dims for i in
                           dimension_system.get_dimensional_dependencies(x, mark_dimensionless=True)]
        canon_expr_units = {i for i in dim_dependencies}

        if not canon_expr_units.issubset(set(canon_dim_units)):
            raise TypeError("There is an invalid character in '%s'" % expr,
                            "the expr must be sympy.physics.unit or number")

        seen = set([])
        canon_dim_units = [i for i in canon_dim_units if not (i in seen or seen.add(i))]

        camat = Matrix(
            [[dimension_system.get_dimensional_dependencies(i, mark_dimensionless=True).get(j, 0) for i in target_dims]
             for j in canon_dim_units])
        exprmat = Matrix([dim_dependencies.get(k, 0) for k in canon_dim_units])

        res_exponents = camat.solve_least_squares(exprmat, method=None)

        return res_exponents, canon_dim_units

    @classmethod
    def convert_to(cls, expr, target_units=None, unit_system="SI"):

        from sympy.physics.units import UnitSystem
        unit_system = UnitSystem.get_unit_system(unit_system)
        if not target_units:
            target_units = unit_system._base_units

        if not isinstance(target_units, (Iterable, Tuple)):
            target_units = [target_units]

        if isinstance(expr, Add):
            raise TypeError("can not be add")

        expr = sympify(expr)

        if not isinstance(expr, Quantity) and expr.has(Quantity):
            expr = expr.replace(lambda x: isinstance(x, Quantity), lambda x: x.convert_to(target_units, unit_system))

        def get_total_scale_factor(expr0):
            if isinstance(expr0, Mul):
                return reduce(lambda x, y: x * y, [get_total_scale_factor(i) for i in expr0.args])
            elif isinstance(expr0, Pow):
                return get_total_scale_factor(expr0.base) ** expr0.exp
            elif isinstance(expr0, Quantity):
                return unit_system.get_quantity_scale_factor(expr0)
            return expr0

        depmat, canon_dim_units = cls._get_conversion_matrix_for_expr(expr, target_units, unit_system)
        if depmat is None:
            raise TypeError("There is an invalid character in '%s'" % expr,
                            "the expr must be sympy.physics.unit or number")

        expr_scale_factor = get_total_scale_factor(expr)
        dim_dict = {}
        for u, p in zip(target_units, depmat):
            expr_scale_factor /= get_total_scale_factor(u) ** p
            dim_dict["%s" % u] = p

        d = cls(np.array(list(dim_dict.values())))
        d.dim = canon_dim_units
        d.unit = target_units
        return (expr_scale_factor, d)

    @classmethod
    def convert_to_Dim(cls, u, target_units=None, unit_system="SI"):
        """

        Parameters
        ----------
        u: sympy.physics.unit or Expr of sympy.physics.unit
        target_units: None or list of sympy.physics.unit
            if None, the target_units is 7 SI units
        unit_system: str
            default is unit_system="SI"
        Returns
        -------
        Expr
        """
        if isinstance(u, Dim):
            return 1, u
        else:
            expr_scale_factor, d = cls.convert_to(u, target_units=target_units, unit_system=unit_system)
            return expr_scale_factor, d

    @classmethod
    def convert_x(cls, xi, ui, target_units=None, unit_system="SI"):
        """
        Quick method. translate xi and ui to standard system.
        Parameters
        ----------
        xi: np.ndarray
        ui: sympy.physics.unit or Expr of sympy.physics.unit
        target_units: None or list of sympy.physics.unit
            if None, the target_units is 7 SI units
        unit_system: str
            default is unit_system="SI"

        Returns
        -------
        xi: np.ndarray
        expr: Expr
        """
        expr_scale_factor, d = cls.convert_to_Dim(ui, target_units=target_units, unit_system=unit_system)
        return expr_scale_factor * xi, d

    @classmethod
    def inverse_convert(cls, dim, scale=1, target_units=None, unit_system="SI"):
        """
        Quick method. Translate ui to other unit.
        Parameters
        ----------
        dim: Dim
        scale: float
        target_units: None or list of sympy.physics.unit
            if None, the target_units is 7 SI units
        unit_system: str
            default is unit_system="SI"

        Returns
        -------
        scale: float
        expr: Expr
        """
        from sympy.physics.units import UnitSystem
        unit_system = UnitSystem.get_unit_system(unit_system)
        if not target_units:
            target_units = unit_system._base_units

        if not isinstance(target_units, (Iterable, Tuple)):
            target_units = [target_units]

        def get_total_scale_factor(expr):
            if isinstance(expr, Mul):
                return reduce(lambda x, y: x * y, [get_total_scale_factor(i) for i in expr.args])
            elif isinstance(expr, Pow):
                return get_total_scale_factor(expr.base) ** expr.exp
            elif isinstance(expr, Quantity):
                return unit_system.get_quantity_scale_factor(expr)
            return expr

        sc = scale * Mul.fromiter(1 / get_total_scale_factor(u) ** p for u, p in zip(target_units, dim))
        sc = sc * Mul.fromiter(1 / get_total_scale_factor(u) ** p for u, p in zip(unit_system._base_units, dim))
        tar = Mul.fromiter((1 / get_total_scale_factor(u) * u) ** p for u, p in zip(target_units, dim))
        bas = Mul.fromiter((get_total_scale_factor(u)) ** p for u, p in zip(unit_system._base_units, dim))
        return sc, scale * tar * bas

    @classmethod
    def inverse_convert_x(cls, xi, dim, scale=1, target_units=None, unit_system="SI"):
        """
        Quick method. Translate xi, dim to other unit.
        Parameters
        ----------
        xi:np.ndarray
        dim: Dim
        scale: float
            if xi is have been scaled, the scale is 1.
        target_units: None or list of sympy.physics.unit
            if None, the target_units is 7 SI units
        unit_system: str
            default is unit_system="SI"

        Returns
        -------
        scale: float
        expr: Expr
        """
        expr_scale_factor, d = cls.inverse_convert(dim, scale=scale, target_units=target_units, unit_system=unit_system)
        return expr_scale_factor * xi, d


dnan = Dim(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]))
dless = Dim(np.array([0, 0, 0, 0, 0, 0, 0]))


