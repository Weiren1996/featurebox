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
from joblib import Parallel, delayed, effective_n_jobs
from tqdm import tqdm


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


def parallize(n_jobs, func, iterable, respective=False, **kwargs):
    """
    parallize the function for iterable.
    use in if __name__ == "__main__":

    Parameters
    ----------
    respective
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
    if respective:
        return parallel(func(*iter_i) for iter_i in tqdm(iterable))
    else:
        return parallel(func(iter_i) for iter_i in tqdm(iterable))


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


def name_to_name(*iters, search=None, search_which=1, return_which=(1,), two_layer=False):
    if isinstance(return_which, int):
        return_which = tuple([return_which, ])
    if two_layer:

        results_all = []
        if isinstance(search, Iterable):
            for index_i in search:
                results_all.append(
                    name_to_name(*iters, search=index_i, search_which=search_which,
                                 return_which=return_which, two_layer=False))

            if len(return_which) >= 2:
                return list(zip(*results_all))
            else:
                return results_all
        else:
            raise IndexError("search_name or search should be iterable")

    else:

        zeros = [list(range(len(iters[0])))]

        zeros.extend([list(_) for _ in iters])

        iters = zeros

        zips = list(zip(*iters))

        if isinstance(search, Iterable):

            search_index = [iters[search_which].index(i) for i in search]

            results = [zips[i] for i in search_index]

        else:

            raise IndexError("search_name or search should be iterable")

        res = list(zip(*results))

        if not res:
            return_res = [[] for _ in return_which]
        else:
            return_res = [res[_] for _ in return_which]

        if len(return_which) == 1:
            return_res = return_res[0]
        return return_res


list1 = [1, 2, 3, 4, 5]
list2 = ["a", "b", "c", "d", "e"]

# a = name_to_name(list(range(len(list1))),list2, search=["a","e"], search_which=2, return_which=(0,1), two_layer=False)
a = name_to_name(list1, list2, search=[["a", "e"], ["b", "e"]], search_which=2, return_which=1, two_layer=True)
