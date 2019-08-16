# -*- coding: utf-8 -*-

# @Time   : 2019/5/30 10:54
# @Author : Administrator
# @Project : feature_preparation
# @FileName: cuplefeature.py
# @Software: PyCharm


"""
this is a description
"""
from abc import ABC
import numpy as np

from featurebox.featurizer.basefeaturizer import BaseFeaturizer


class ElementCupleFeaturizer(BaseFeaturizer, ABC):

    def __init__(self, comp, elem_data, couple=2, stats=("mean",)):
        self.couple = couple
        self.comp = comp
        self.elem_data = elem_data
        self.stats = stats
        # Initialize stats computer

    def featurize(self, comp_number=0):
        """
        Get elemental property attributes

        Args:
            comp: Pymatgen composition object

        Returns:
            all_attributes: Specified property statistics of features
            :param comp_number:
        """
        comp = self.comp[comp_number]
        elem_data = self.elem_data[comp_number].values
        all_attributes = []

        # Get the element names and fractions
        elements, fractions = zip(*comp.element_composition.items())
        elem_data = np.reshape(elem_data, (self.couple, -1), order="F")
        for elem_data_i in elem_data.T:
            for stat in self.stats:
                all_attributes.append(self.pstats.calc_stat(elem_data_i, stat, fractions))

        return all_attributes

    def transform(self, X=None):
        """Compute features for a list of inputs"""
        number = list(range(len(self.comp)))
        return self.featurize_many(number, ignore_errors=True)

    def feature_labels(self):
        """
        Generate attribute names.

        Returns:
            ([str]) attribute labels.
        """
        name = np.array(self.elem_data.columns.values)[::self.couple]
        name = [i.split("_")[0] for i in name]
        return name
