from sklearn.base import BaseEstimator, TransformerMixin, MultiOutputMixin
from sklearn.metrics import check_scoring

from featurebox.symbol import flow
from featurebox.symbol.base import SymbolSet
from featurebox.symbol.calculation.scores import calculate_y_unpack
from featurebox.symbol.calculation.translate import general_expr
from featurebox.symbol.flow import MutilMutateLoop
from featurebox.symbol.functions.dimfunc import Dim


class SymbolLearning(BaseEstimator, MultiOutputMixin, TransformerMixin):
    """One simplify API for flow.\n
    The detailed functions should turn to the loop of featurebox.symbol.flow.\n
    The SymbolLearning is time costing and are not suit for GridSearchCV, the cross_validate are builtin """

    def __str__(self):
        return str(self.loop)

    def __init__(self, *args, loop=None, **kwargs):
        """
        Parameters
        ----------
        pset:SymbolSet
            the feature x and traget y and others should have been added.
        loop: str
            featurebox.symbol.flow.BaseLoop
            [“BaseLoop”,”MutilMutateLoop“,“OnePointMutateLoop”, ”DimForceLoop“...]
        pop:int
            number of popolation
        gen:int
            number of generation
        mutate_prob:float
            probability of mutate
        mate_prob:float
            probability of mate(crossover)
        initial_max:int
            max initial size of expression when first producing.
        initial_min : None,int
            max initial size of expression when first producing.
        max_value:int
            max size of expression
        hall:int,>=1
            number of HallOfFame(elite) to maintain
        re_hall:None or int>=2
            Notes: only vaild when hall
            number of HallOfFame to add to next generation.
        re_Tree: int
            number of new features to add to next generation.
            0 is false to add.
        personal_map:bool or "auto"
            "auto" is using premap and with auto refresh the premap with individual.\n
            True is just using constant premap.\n
            False is just use the prob of terminals.
        scoring: list of Callbale, default is [sklearn.metrics.r2_score,]
            See Also sklearn.metrics
        score_pen: tuple of  1, -1 or float but 0.
            >0 : max problem, best is positive, worse -np.inf
            <0 : min problem, best is negative, worse np.inf
            Notes:
            if multiply score method, the scores must be turn to same dimension in preprocessing
            or weight by score_pen. Because the all the selection are stand on the mean(w_i*score_i)
            Examples: [r2_score] is [1],
        cv:sklearn.model_selection._split._BaseKFold,int
            the shuffler must be False,
            default=1 means no cv
        filter_warning:bool
            filter warning or not
        add_coef:bool
            add coef in expression or not.
        inter_add：bool
            add intercept constant or not
        inner_add:bool
            dd inner coeffcients or not
        n_jobs:int
            default 1, advise 6
        batch_size:int
            default 40, depend of machine
        random_state:int
            None,int
        cal_dim:bool
            excape the dim calculation
        dim_type:Dim or None or list of Dim
            "coef": af(x)+b. a,b have dimension,f(x) is not dnan. \n
            "integer": af(x)+b. f(x) is interger dimension. \n
            [Dim1,Dim2]: f(x) in list. \n
            Dim: f(x) ~= Dim. (see fuzzy) \n
            Dim: f(x) == Dim. \n
            None: f(x) == pset.y_dim
        fuzzy:bool
            choose the dim with same base with dim_type,such as m,m^2,m^3.
        stats:dict
            details of logbook to show. \n
            Map:\n
            values
                = {"max": np.max, "mean": np.mean, "min": np.mean, "std": np.std, "sum": np.sum}
            keys
                = {\n
                   "fitness": just see fitness[0], \n
                   "fitness_dim_max": max problem, see fitness with demand dim,\n
                   "fitness_dim_min": min problem, see fitness with demand dim,\n
                   "dim_is_target": demand dim,\n
                   "coef":  dim is true, coef have dim, \n
                   "integer":  dim is integer, \n
                   ...
                   }
            if stats is None, default is :\n
                stats = {"fitness_dim_max": ("max",), "dim_is_target": ("sum",)}   for cal_dim=True
                stats = {"fitness": ("max",)}                                      for cal_dim=False
            if self-definition, the key is func to get attribute of each ind./n
            Examples:
                def func(ind):\n
                    return ind.fitness[0]
                stats = {func: ("mean",), "dim_is_target": ("sum",)}
        verbose:bool
            print verbose logbook or not
        tq:bool
            print progress bar or not
        store:bool or path
            bool or path
        stop_condition:callable
            stop condition on the best ind of hall, which return bool,the true means stop loop.
            Examples:
                def func(ind):\n
                    c = ind.fitness.values[0]>=0.90
                    return c
        """
        self.args = args
        self.kwargs = kwargs
        if loop is None:
            loop = MutilMutateLoop
        if isinstance(loop, str):
            loop = getattr(flow, loop)

        self.loop = loop

    def fit(self, X=None, y=None, c=None, x_group=None, pset=None):
        """

        If pset is None, one simple pset are generate with no dimension calculation, But just with x_group.\n
        If need more self-definition, use one defined SymbolSet object to pset.\n
        Examples:
            pset = SymbolSet()\n
            pset.add_features_and_constants(...)\n
            pset.add_operations(...)\n
            ...\n
            ...SymbolLearning().fit(pset=pset)\n

        Parameters
        ----------
        X:np.ndarray
        
        y:np.ndarray
        
        c:list of float
        
        x_group:list of list
            Group of x.\n
            See Also pset.add_features_and_constants
        pset:SymbolSet
            See Also SymbolSet

        """
        if pset is None:
            pset = SymbolSet()
            pset.add_features_and_constants(X, y, c, x_dim=1, y_dim=1, c_dim=1, x_prob=None,
                                            c_prob=None, x_group=x_group, feature_name=None)
            pset.add_operations(power_categories=(2, 3, 0.5),
                                categories=("Add", "Mul", "Sub", "Div"))

        self.loop = self.loop(pset, *self.args, **self.kwargs)
        hall = self.loop.run()
        self.best_one = hall.items[0]
        try:
            expr = general_expr(self.best_one.coef_expr, self.loop.cpset)
            self.expr_type = "single"
        except (RecursionError, RuntimeWarning):
            expr = self.best_one.coef_expr
            self.expr_type = "group"

        self.expr = expr
        self.y_dim = self.best_one.y_dim
        self.fitness = self.best_one.fitness.values[0]

    def _predict_by_single(self, X):

        terminals = self.loop.cpset.init_free_symbol
        indexs = [int(i.name.replace("x", "")) for i in terminals if "x" in i.name]
        X = [xi for xi in X.T]
        X = [X[indexi] for indexi in indexs]

        c = []
        for i in self.loop.cpset.data_x_dict.keys():
            if "c" in i:
                c.append(self.loop.cpset.data_x_dict[i])
        X_and_c = X + c
        pre_y = calculate_y_unpack(self.expr, X_and_c, terminals)
        return pre_y

    def _predict_by_group(self, X):
        from copy import deepcopy
        cpset_new = deepcopy(self.loop.cpset)
        se = cpset_new.replace(X)

        res = se.calculate_score(self.expr)
        score, expr01, pre_y = res
        return pre_y

    def predict(self, X):
        if self.expr_type == "group":
            return self._predict_by_group(X)
        else:
            return self._predict_by_single(X)

    def score(self, X, y, scoring):

        y_pred = self.predict(X)

        scoring = check_scoring(self, scoring=scoring)

        if not isinstance(scoring, (list, tuple)):
            scoring = [scoring, ]
        try:
            sc_all = []
            for si in scoring:
                sc = si(self, y, y_pred)
                sc_all.append(sc)

        except (ValueError, RuntimeWarning):

            sc_all = None

        return sc_all

    def cv_result(self, refit=True):
        if self.loop.cpset.cv != 1:
            self.loop.cpset.refit = refit
            return self.loop.cpset.calculate_cv_score(self.best_one.expr)
        else:
            return None


if __name__ == "__main__":
    # data
    from sklearn.datasets import load_boston

    data = load_boston()
    x = data["data"]
    y = data["target"]
    c = [6, 3, 4]

    sl = SymbolLearning(pop=50, gen=2, cal_dim=True, re_hall=2, add_coef=True, cv=2, random_state=0
                        )
    sl.fit(x, y, c=c, x_group=[[1, 3], [0, 2], [4, 7]])
    score = sl.score(x, y, "r2")
