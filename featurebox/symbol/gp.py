#!/usr/bin/python
# coding:utf-8

"""
@author: wangchangxin
@contact: 986798607@qq.com
@software: PyCharm
@file: symbol.py
@time: 2020/5/16 14:46
"""
import sys
from inspect import isclass
from numpy import random


def generate(pset, min_, max_, condition, *kwargs):
    """

    Parameters
    ----------
    pset: SymbolSet
    min_: int
        Minimum height of the produced trees.
    max_: int
        Maximum Height of the produced trees.
    condition: collections.Callable
        The condition is a function that takes two arguments,
        the height of the tree to build and the current
        depth in the tree.
    kwargs: None
        placeholder for future

    Returns
    -------

    """
    _ = kwargs
    type_ = object
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth):
            try:
                term = random.choice(pset.terminals + pset.constants, p=pset.prob_pro_ter_con_list)
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The symbol.generate function tried to add "
                                 "a terminalm, but there is "
                                 "none available.").with_traceback(traceback)
            if isclass(term):
                term = term()
            expr.append(term)
        else:
            try:
                prim = random.choice(pset.primitives, p=pset.prob_pri_list)
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The symbol.generate function tried to add "
                                 "a primitive', but there is "
                                 "none available.").with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))

    dispose = list(random.choice(pset.dispose, len(expr), p=pset.prob_dispose_list))
    dispose[0] = pset.dispose[1]

    re = []
    for i, j in zip(dispose, expr):
        re.extend((i, j))

    return re


def genGrow(pset, min_, max_):
    """Generate an expression where each leaf might have a different depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :returns: A grown tree with leaves at possibly different depths.
    """

    def condition(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a node should be a terminal.
        """
        return depth == height or (depth >= min_ and random.random() < pset.terminalRatio)

    return generate(pset, min_, max_, condition)


def compile_(expr, pset):
    """Compile the expression *expr*.

    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param pset: Primitive set against which the expression is compile.
    :returns: a function if the primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.
    """
    if isinstance(expr, str):
        code = expr
    else:
        code = repr(expr)
    if len(pset.arguments) > 0:
        # This section is a stripped version of the lambdify
        # function of SymPy 0.6.6.
        args = ",".join(arg for arg in pset.arguments)
        code = "lambda {args}: {code}".format(args=args, code=code)
    try:
        return eval(code, pset.context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError("DEAP : Error in tree evaluation :"
                          " Python cannot evaluate a tree higher than 90. "
                          "To avoid this problem, you should use bloat control on your "
                          "operators. See the DEAP documentation for more information. "
                          "DEAP will now abort.").with_traceback(traceback)
