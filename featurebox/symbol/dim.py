#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Time    : 2019/11/12 15:13
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: GNU Lesser General Public License v3.0

"""
Notes:
    this some of this part are a copy from sympy
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


# from sympy.physics.units.systems import SI
# from featurebox.tools.tool import parallelize


def dim_map():
    """expr to dim function """

    def my_comp(dim):

        if isinstance(dim, Dim):
            if dim.ndim == 1:
                return dim
            else:
                n = dim.shape[0]
                dc = dim[0].copy()
                return dc.__pow__(n)
        else:
            return dless

    def my_quot(dim):
        if isinstance(dim, Dim):
            if dim.ndim == 1:
                return dim
            elif dim.shape[0] == 2:
                return dless
            else:
                return dim
        else:
            return dless

    def my_diff(dim):
        if isinstance(dim, Dim):
            if dim.ndim == 1:
                return dim
            elif dim.shape[0] == 2:
                return dim[0].copy()
            else:
                return dim
        else:
            return dless

    def my_flat(dim):
        if isinstance(dim, Dim):
            if dim.ndim == 1:
                return dim
            else:
                return dim[0].copy()
        else:
            return dless

    def my_abs(dim):
        if isinstance(dim, Dim):
            return dim
        else:
            return dless

    def my_sqrt(dim):

        return dim.__pow__(0.5)

    my_self = my_abs

    def my_exp(dim):

        if isinstance(dim, Dim):
            if dim == dless:
                return dless.get_n(dim)
            else:
                return dnan.get_n(dim)
        else:
            return dless

    my_log = my_cos = my_sin = my_exp

    my_funcs = {"Abs": my_abs, "exp": my_exp, "log": my_log, 'cos': my_cos, 'sin': my_sin,
                'sqrt': my_sqrt, "MAdd": my_flat, "MMul": my_comp, "MSub": my_diff,
                "MDiv": my_quot, "Self": my_self}
    return my_funcs


class Dim(numeric.ndarray):
    """Redefine the Dimension of sympy, the default dimension SI system with 7 number,
    1.can be constructed by list of number.
    1.can be translated from a sympy.physics.unit.
        # >>>from sympy.physics.units import N
        # >>>scale,dim = Dim.convert_to_Dim(N)
        #inverse back
    """

    def __new__(cls, data):

        assert isinstance(data, (numeric.ndarray, list))
        dtype = np.float16

        arr = numeric.array(data, dtype=dtype, copy=True)

        shape = arr.shape

        ret = numeric.ndarray.__new__(cls, shape, dtype=np.float16,
                                      buffer=arr,
                                      order='c')
        return ret

    def __init__(self, _):
        """
        # self.unit = [str(i) for i in SI._base_units]
        # self.unit_map = {'meter': "m", 'kilogram': "kg", 'second': "s",
        #                  'ampere': "A", 'mole': "mol", 'candela': "cd", 'kelvin': "K"}
        # self.dim = ['length', 'mass', 'time', 'current', 'amount_of_substance',
        #             'luminous_intensity', 'temperature']
        """

    def __eq__(self, other):
        se = self.copy()
        ot = other.copy()
        if se.ndim == 2:
            se = se[0]
        if ot.ndim == 2:
            ot = ot[0]
        return all(np.equal(se, ot))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):

        if isinstance(other, Dim) and self != other:
            if other == dless:
                return self.get_n(other)
            elif self == dless:
                return other.get_n(self)
            else:
                return dnan.get_n(self.get_n(other))  # ?
        elif isinstance(other, Dim) and self == other:
            return self.get_n(other)

        else:
            return self

    def __sub__(self, other):
        return self.__add__(other)

    def __pow__(self, other):
        return self._eval_power(other)

    def _eval_power(self, other):
        if isinstance(other, (numbers.Real, sympy.Rational, sympy.Float)):
            return Dim(np.array(self) * other)
        else:
            return dnan  #

    def __mul__(self, other):
        if isinstance(other, Dim):
            return Dim(np.array(self) + np.array(other))
        else:
            return self

    def __div__(self, other):

        if isinstance(other, Dim):
            return Dim(np.array(self) - np.array(other))
        else:
            return self

    def __rdiv__(self, other):

        if isinstance(other, (numbers.Real, sympy.Rational, sympy.Float)):
            return self.__pow__(-1)
        else:
            return dnan  #

    def __abs__(self):
        return self

    def __rpow__(self, other):
        return dnan  #

    def __neg__(self):
        return self

    def __pos__(self):
        return self

    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__

    # @property
    def allisnan(self):
        return all(np.isnan(self))

    # @property
    def anyisnan(self):
        return any(np.isnan(self))

    # @property
    def isfloat(self):
        return any(np.modf(self)[0])

    # @property
    def isinteger(self):
        return not any(np.modf(self)[0])

    def is_same_base(self, others):
        se = self.copy()
        if others.ndim == 2:
            others = others[0]
        if se.ndim == 2:
            se = se[0]

        if isinstance(others, Dim):
            npself = np.array(se)
            npothers = np.array(others)
            x1 = np.linalg.norm(npself)
            x2 = np.linalg.norm(npothers)

            if others ** x1 == se ** x2:
                return True
            else:
                return False
        else:
            return False

    def get_n(self, others):
        se = self.copy()

        if others.ndim == 2:
            n = others.shape[0]
        else:
            n = 1

        if self.ndim == 2:
            m = self.shape[0]
        else:
            m = 1

        if m > 1 and n > 1:
            return dnan  #
        elif n == 1:
            return se
        elif m == 1 and n > 1:
            return Dim(np.array([se] * n))
        else:
            return dnan  #

    @staticmethod
    def _get_conversion_matrix_for_expr(expr, target_units, dimsys_default=None):
        from sympy import Matrix
        if not dimsys_default:
            from sympy.physics.units.dimensions import dimsys_default

        expr_dim = Dimension(Quantity.get_dimensional_expr(expr))
        dim_dependencies = dimsys_default.get_dimensional_dependencies(expr_dim, mark_dimensionless=True)
        target_dims = [Dimension(Quantity.get_dimensional_expr(x)) for x in target_units]

        canon_dim_units = {i for x in target_dims for i in
                           dimsys_default.get_dimensional_dependencies(x, mark_dimensionless=True)}
        canon_expr_units = {i for i in dim_dependencies}

        if not canon_expr_units.issubset(canon_dim_units):
            return None

        canon_dim_units = sorted(canon_dim_units)

        camat = Matrix(
            [[dimsys_default.get_dimensional_dependencies(i, mark_dimensionless=True).get(j, 0) for i in target_dims]
             for j in canon_dim_units])
        exprmat = Matrix([dim_dependencies.get(k, 0) for k in canon_dim_units])

        res_exponents = camat.solve_least_squares(exprmat, method=None)
        return res_exponents, canon_dim_units

    @classmethod
    def convert_to(cls, expr, target_units, unit_system):

        from sympy.physics.units.systems.mks import MKS, _mks_dim
        from sympy.physics.units.systems.mksa import MKSA, _mksa_dim
        from sympy.physics.units.systems.natural import _natural_dim, natural
        from sympy.physics.units.systems.si import SI, _si_dim

        if unit_system == "SI":
            unit_system, dimsys = SI, _si_dim
        elif unit_system == "MKSA":
            unit_system, dimsys = MKSA, _mksa_dim
        elif unit_system == "MKS":
            unit_system, dimsys = MKS, _mks_dim
        elif unit_system == "natural":
            unit_system, dimsys = natural, _natural_dim
        else:
            raise NameError('only,"SI","MKSA","MKS","natural" is definition')

        if not target_units:
            target_units = unit_system._base_units
        if not isinstance(target_units, (Iterable, Tuple)):
            target_units = [target_units]

        if isinstance(expr, Add):
            return Add.fromiter(cls.convert_to(i, target_units) for i in expr.args)

        expr = sympify(expr)

        if not isinstance(expr, Quantity) and expr.has(Quantity):
            expr = expr.replace(lambda x: isinstance(x, Quantity), lambda x: x.convert_to(target_units))

        def get_total_scale_factor(expr):
            if isinstance(expr, Mul):
                return reduce(lambda x, y: x * y, [get_total_scale_factor(i) for i in expr.args])
            elif isinstance(expr, Pow):
                return get_total_scale_factor(expr.base) ** expr.exp
            elif isinstance(expr, Quantity):
                return expr.scale_factor
            return expr

        depmat, canon_dim_units = cls._get_conversion_matrix_for_expr(expr, target_units, dimsys)

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
    def convert_xi(cls, xi, ui, target_units=None, unit_system="SI"):
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
    def convert_x(cls, x, u, target_units=None, unit_system="SI"):
        """
         Quick method. translate x and u to standard system.
         Parameters
         ----------
         x: np.ndarray or list of ndarray
         u: list of sympy.physics.unit or Expr of sympy.physics.unit
         target_units: None or list of sympy.physics.unit
             if None, the target_units is 7 SI units
         unit_system: str
             default is unit_system="SI"

         Returns
         -------
         x: np.ndarray
         expr: Expr
         """
        if isinstance(x, list):
            pass
        elif isinstance(x, np.ndarray):
            x = x.T
        else:
            raise TypeError("values must be list or np.array")
        x_and_d = [cls.convert_xi(xi, x_ui, target_units, unit_system) for xi, x_ui in zip(x, u)]
        x = np.array([xi for xi, d in x_and_d]).T
        x_dim = [d for xi, d in x_and_d]
        return x, x_dim

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
        from sympy.physics.units.systems.mks import MKS, _mks_dim
        from sympy.physics.units.systems.mksa import MKSA, _mksa_dim
        from sympy.physics.units.systems.natural import _natural_dim, natural
        from sympy.physics.units.systems.si import SI, _si_dim

        if unit_system == "SI":
            unit_system, dimsys = SI, _si_dim
        elif unit_system == "MKSA":
            unit_system, dimsys = MKSA, _mksa_dim
        elif unit_system == "MKS":
            unit_system, dimsys = MKS, _mks_dim
        elif unit_system == "natural":
            unit_system, dimsys = natural, _natural_dim
        else:
            raise NameError('only,"SI","MKSA","MKS","natural" is definition')

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
    def inverse_convert_xi(cls, xi, dim, scale=1, target_units=None, unit_system="SI"):
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


dnan = Dim(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]))
dless = Dim(np.array([0, 0, 0, 0, 0, 0, 0]))

if __name__ == "__main__":
    a = [Dim([1, 2, 3, 4, 5, 6, 7])] * 100
    b = [Dim([2, 2, 3, 4, 5, 6, 7])] * 100

    dl = dless
    dn = dnan

    c = Dim([1, 2, 3, 4, 5, 6, 7])

    t = zip(a, b)


    def func(k):
        ss = k[1] * k[0]
        return ss
