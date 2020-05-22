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
from collections.abc import Iterable
from functools import partial, wraps

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from tqdm import tqdm


def time_this_function(func):
    """
    time the function. use as a decorator.

    Parameters
    ----------
    func: Callable
        function

    Returns
    -------
    result
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
    seed: None,int or RandomState
        If seed is None, return the RandomState singleton used by random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    random.Random
        RandomState object
    """

    if seed is None or seed is random.random:
        return random.Random()
    if isinstance(seed, (numbers.Integral, np.integer)):
        return random.Random(seed)
    if isinstance(seed, random.Random):
        return seed
    raise ValueError('%r cannot be used to seed a seed'
                     ' instance' % seed)


def parallelize(n_jobs, func, iterable, respective=False, tq=True, batch_size='auto', **kwargs):
    """
    parallelize the function for iterable.

    make sure in if __name__ == "__main__":

    Parameters
    ----------
    respective:bool
        Import the parameters respectively or as a whole
    tq:bool
         View Progress or not
    n_jobs:int
        cpu numbers. n_jobs is the number of workers requested by the callers. Passing n_jobs=-1
    means requesting all available workers for instance matching the number of CPU cores on the worker host(s).
    func:
        function to calculate
    iterable:
        interable object
    kwargs:
        kwargs for function

    Returns
    -------
    results
        function results

    """

    func = partial(func, **kwargs)
    if effective_n_jobs(n_jobs) == 1:
        parallel, func = list, func
    else:
        parallel = Parallel(n_jobs=n_jobs, batch_size=batch_size)
        func = delayed(func)
    if tq:
        if respective:
            return parallel(func(*iter_i) for iter_i in tqdm(iterable))
        else:
            return parallel(func(iter_i) for iter_i in tqdm(iterable))
    else:
        if respective:
            return parallel(func(*iter_i) for iter_i in iterable)
        else:
            return parallel(func(iter_i) for iter_i in iterable)


def logg(func, printing=True, back=False):
    """
    Get the name of function

    Parameters
    ----------
    func:Callable
        function to calculate
    printing:bool
        print or not
    back:bool
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
        if printing:
            print(name, arg_dict)
        if back:
            return (name, arg_dict), result
        else:
            return result

    return wrapper


def name_to_name(*iters, search, search_which=1, return_which=(1,), two_layer=False):
    """

    Parameters
    ----------
    iters:Iterable
        iterable objects to select and sort
    search:Iterable
        the rank basis
    search_which:int
        the (the index of iters) of the rank basis. where to rank
    return_which:tuple of int
        return index in inters
    two_layer:bool
        search is nested with two layer or not

    Returns
    -------
    result:Iterable
        Result of find and sort
    """
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


if __name__ == "__main__":
    list1 = [1, 2, 3, 4, 5]
    list2 = ["a", "b", "c", "d", "e"]

    # a = name_to_name(list(range(len(list1))),list2, search=["a","e"], search_which=2,
    # return_which=(0,1), two_layer=False)
    a = name_to_name(list1, list2, search=[["a", "e"], ["b", "e"]], search_which=2, return_which=1, two_layer=True)
