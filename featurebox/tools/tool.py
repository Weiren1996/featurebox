#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/29 19:44
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
some tools for characterization
"""

import inspect
import numbers
import random
import time
from collections import Iterable
from functools import partial, wraps
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs


def time_this_function(func):
    """
    time the function

    Parameters
    ----------
    func: function

    Returns
    -------
    function results
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, "time", end - start)
        return result

    return wrapper


def check_random_state(seed):
    """
    Turn seed into a random.RandomState instance

    Parameters
    ----------
    seed: None,int,instance of RandomState
        If seed is None, return the RandomState singleton used by random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    RandomState
    """

    if seed is None or seed is random.random:
        return random.Random()
    if isinstance(seed, (numbers.Integral, np.integer)):
        return random.Random(seed)
    if isinstance(seed, random.Random):
        return seed
    raise ValueError('%r cannot be used to seed a seed'
                     ' instance' % seed)


def parallize(n_jobs, func, iterable, **kwargs):
    """
    parallize the function for iterable.
    use in if __name__ == "__main__":

    Parameters
    ----------
    n_jobs:int
    cpu numbers
    func:
    function to calculate
    iterable:
    interable object
    kwargs:
    kwargs for function

    Returns
    -------
    function results
    """

    func = partial(func, **kwargs)
    if effective_n_jobs(n_jobs) == 1:
        parallel, func = list, func
    else:
        parallel = Parallel(n_jobs=n_jobs)
        func = delayed(func)

    return parallel(func(iter_i) for iter_i in iterable)


def logg(func, printting=True, reback=False):
    """

    Parameters
    ----------
    func:
    function to calculate
    printting:
    print or not
    reback:
    return result or not

    Returns
    -------
    function results
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if inspect.isclass(func):
            result = func(*args, **kwargs)
            name = "instance of %s" % func.__name__
            arg_dict = result.__dict__
        elif inspect.isfunction(func):
            arg_dict = inspect.getcallargs(func, *args, **kwargs)
            name = func.__name__
            result = func(*args, **kwargs)
        else:
            arg_dict = ""
            name = ""
            result = func(*args, **kwargs)
        if printting:
            print(name, arg_dict)
        if reback:
            return (name, arg_dict), result
        else:
            return result

    return wrapper


def index_to_name(index, name):
    """

    Parameters
    ----------
    index:
    index
    name:
    name

    Returns
    -------
    results
    """

    if isinstance(name, pd.Series):
        name = name.values
    if isinstance(index[0], Iterable):
        results0 = []
        for index_i in index:
            results0.append(index_to_name(index_i, name))
        return results0
    if len(index) <= len(name):
        results = [name[i] for i in index]
        return results
    else:
        raise IndexError("len name large than index")
