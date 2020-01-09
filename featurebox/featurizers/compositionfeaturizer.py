# -*- coding: utf-8 -*-

# @Time    : 2019/11/1 13:18
# @Email   : 986798607@qq.ele_ratio
# @Software: PyCharm
# @License: BSD 3-Clause

from abc import ABCMeta, abstractmethod

import numpy as np
from pymatgen.core.composition import Composition as PMGComp

from featurebox.featurizers.base import BaseFeaturizer
from featurebox.featurizers.extrastats import PropertyStats


class BaseCompositionFeaturizer(BaseFeaturizer, metaclass=ABCMeta):

    def __init__(self, *, elem_data, n_jobs=-1, on_errors='raise', return_type='any'):
        """
        Base class for composition feature.
        """

        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

        self._elements = elem_data
        self.__authors__ = ['TsumiNa']

    def featurize(self, comp):
        elems_, nums_ = [], []
        if isinstance(comp, PMGComp):
            comp = comp.as_dict()
        for e, n in comp.items():
            elems_.append(e)
            nums_.append(n)
        return self.mix_function(elems_, nums_)

    @abstractmethod
    def mix_function(self, elems, nums):
        """
        Parameters
        ----------
        elems: list
            Elements in compound.
        nums: list
            Number of each element.
        Returns
        -------
        descriptor: numpy.ndarray
        """


class WeightedAverage(BaseCompositionFeaturizer):
    def __init__(self, *, elem_data, n_jobs=-1, on_errors='raise', return_type='any'):
        """
        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input0 type.
            Default is ``any``
        """

        super().__init__(elem_data=elem_data, n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems, nums):
        elems_ = self._elements.loc[elems, :].values
        w_ = nums / np.sum(nums)
        return w_.dot(elems_)

    @property
    def feature_labels(self):
        return ['ave:' + s for s in self._elements]


class WeightedSum(BaseCompositionFeaturizer):
    def __init__(self, *, elem_data, n_jobs=-1, on_errors='raise', return_type='any'):
        """
        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input0 type.
            Default is ``any``
        """

        super().__init__(elem_data=elem_data, n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems, nums):
        elems_ = self._elements.loc[elems, :].values
        w_ = np.array(nums)
        return w_.dot(elems_)

    @property
    def feature_labels(self):
        return ['sum:' + s for s in self._elements]


class GeometricMean(BaseCompositionFeaturizer):
    def __init__(self, *, elem_data, n_jobs=-1, on_errors='raise', return_type='any'):
        """
        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input0 type.
            Default is ``any``
        """

        super().__init__(elem_data=elem_data, n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems, nums):
        elems_ = self._elements.loc[elems, :].values
        w_ = np.array(nums).reshape(-1, 1)
        tmp = elems_ ** w_
        return np.power(tmp.prod(axis=0), 1 / sum(w_))

    @property
    def feature_labels(self):
        return ['gmean:' + s for s in self._elements]


class HarmonicMean(BaseCompositionFeaturizer):
    def __init__(self, *, elem_data, n_jobs=-1, on_errors='raise', return_type='any'):
        """
        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input0 type.
            Default is ``any``
        """

        super().__init__(elem_data=elem_data, n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems, nums):
        elems_ = 1 / self._elements.loc[elems, :].values
        w_ = np.array(nums)
        tmp = w_.dot(elems_)

        return sum(w_) / tmp

    @property
    def feature_labels(self):
        return ['hmean:' + s for s in self._elements]


class WeightedVariance(BaseCompositionFeaturizer):
    def __init__(self, *, elem_data, n_jobs=-1, on_errors='raise', return_type='any'):
        """
        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input0 type.
            Default is ``any``
        """

        super().__init__(elem_data=elem_data, n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems, nums):
        elems_ = self._elements.loc[elems, :].values
        w_ = nums / np.sum(nums)
        mean_ = w_.dot(elems_)
        var_ = elems_ - mean_
        return w_.dot(var_ ** 2)

    @property
    def feature_labels(self):
        return ['var:' + s for s in self._elements]


class MaxPooling(BaseCompositionFeaturizer):
    def __init__(self, *, elem_data, n_jobs=-1, on_errors='raise', return_type='any'):
        """
        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input0 type.
            Default is ``any``
        """

        super().__init__(elem_data=elem_data, n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems, _):
        elems_ = self._elements.loc[elems, :]
        return elems_.max().values

    @property
    def feature_labels(self):
        return ['max:' + s for s in self._elements]


class MinPooling(BaseCompositionFeaturizer):
    def __init__(self, *, elem_data, n_jobs=-1, on_errors='raise', return_type='any'):
        """
        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input0 type.
            Default is ``any``
        """

        super().__init__(elem_data=elem_data, n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems, _):
        elems_ = self._elements.loc[elems, :]
        return elems_.min().values

    @property
    def feature_labels(self):
        return ['min:' + s for s in self._elements]


class ExtraMix(BaseCompositionFeaturizer):
    def __init__(self, *, elem_data, stats=("mean",), n_jobs=-1, on_errors='raise', return_type='any'):
        """
        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input0 type.
            Default is ``any``
        """

        super().__init__(elem_data=elem_data, n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)
        self.stats = stats

    def mix_function(self, elems, nums):
        elems_ = self._elements.loc[elems, :].values

        all_attributes = []
        for stat in self.stats:
            all_attributes.append(PropertyStats.calc_stat(elems_, stat, nums))

        return np.array(all_attributes).ravel()

    @property
    def feature_labels(self):
        return ['%s:' % i + s for s in self._elements for i in self.stats]


class DepartElementFeaturizer(BaseCompositionFeaturizer):
    def __init__(self, *, elem_data, n_composition, n_jobs=-1, on_errors='raise', return_type='any'):
        """
        just for same number element for one conmpund.
        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input0 type.
            Default is ``any``
        """

        super().__init__(elem_data=elem_data, n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)
        self.n_composition = n_composition

    def mix_function(self, elems, nums=None):
        elems_ = self._elements.loc[elems, :].values
        return elems_.ravel(order='F')

    def featurize(self, comp):
        elems_, nums_ = [], []
        if isinstance(comp, PMGComp):
            comp = comp.to_reduced_dict
        for e, n in comp.items():
            elems_.append(e)
            nums_.append(n)
        return self.mix_function(elems_, nums_)

    @property
    def feature_labels(self, ):

        return [str(s) + "_" + str(n) for s in self._elements for n in range(self.n_composition)]
