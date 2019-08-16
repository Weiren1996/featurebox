# -*- coding: utf-8 -*-

# @Time   : 2019/5/18 23:35
# @Author : Administrator
# @Project : feature_preparation
# @FileName: elementfeature.py
# @Software: PyCharm


from pymatgen.core.composition import Composition as PMGComp
from featurebox.featurizer.basefeaturizer import BaseFeaturizer


class DepartElementProPFeaturizer(BaseFeaturizer):
    def __init__(self, elements, *, n_composition=2, n_jobs=-1, include=None,
                 exclude=None, on_errors='raise', return_type='df'):
        """
        Parameters
        ----------
        elements: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
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
            If ``any``, the return type dependent on the input type.
            Default is ``any``
        """

        super().__init__(n_jobs=n_jobs,on_errors=on_errors,return_type=return_type)

        self.elements=elements
        self.n_composition = n_composition

    def _func(self, elems, nums=None):
        elems_ = self.elements.loc[elems, :].values
        return elems_.ravel(order='F')

    def featurize(self, comp):
        elems_, nums_ = [], []
        if isinstance(comp, PMGComp):
            comp = comp.to_reduced_dict
        for e, n in comp.items():
            elems_.append(e)
            nums_.append(n)
        return self._func(elems_, nums_)

    @property
    def feature_labels(self,):

        return [str(s)+"_"+str(n) for s in self.elements for n in range(self.n_composition)]

