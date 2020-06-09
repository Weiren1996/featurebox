import sympy
import numpy as np
from sympy import Function, Expr
from sympy.core.function import UndefinedFunction
from sympy.utilities.lambdify import implemented_function

x = sympy.Symbol("x")
y = sympy.Symbol("y")
z=sympy.pi
# z=sympy.Symbol(str(np.array([1,2,3,4])))
# # sympy.Symbol.
# pi = sympy.pi
# sympy.NumberSymbol
# sympy.AlgebraicNumber
# # sympy.RealNumber
exps=2*x+2*(y+z)
# # func = sympy.utilities.lambdify([x,y],exps ,modules=["numpy", "math"])
# # result = func(np.array([1,2,3,4]),np.array([2,2,2,2]))
#
# f = implemented_function(Function("fs"), lambda x: x+np.array([1,2,3,4]))
# exps = f(exps)
# func = sympy.utilities.lambdify([x,y],exps ,modules=["numpy", "math"])
# result = func(np.array([1,2,3,4]),np.array([2,2,2,2]))

class Coef(UndefinedFunction):

    def __new__(cls, name,arr):

        implementation = lambda x: arr * x
        f = super().__new__(cls, name=name,_imp_=staticmethod(implementation))
        f.arr=arr
        return f

    def __repr__(self):
        return str(self.arr)
    def __str__(self):
        return str(self.arr)
    def __eq__(self, other):
        if isinstance(other,Coef):
            return self.arr==other.arr
        else:
            return False

    def __hash__(self):
        return hash((super().class_key(), frozenset(self._kwargs.items())))

# def prefect_print(expr,w,fea):


co = Coef("fs",np.array([1,2,3,4]))
exps = co(exps)

func = sympy.utilities.lambdify([x,y],exps ,modules=["numpy", "math"])
result = func(np.array([1,2,3,4]),np.array([2,2,2,2]))