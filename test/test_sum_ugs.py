# from sklearn.datasets import load_boston
# from sklearn.datasets import load_iris
# from sklearn.datasets import make_regression
# from sklearn.datasets import make_classification
import warnings
from itertools import chain, combinations

import numpy as np

from featurebox.data.datasets import init, add_noise
from featurebox.selection.exhaustion import Exhaustion
from featurebox.selection.sum import SUM
from featurebox.tools.quickmethod import method_pack

# x, y = load_boston(return_X_y=True)
# x, y = load_iris(return_X_y=True)
# x, y = make_regression(100, 10)
# x, y = make_classification(100, 10)


x = init(100, 10)

warnings.filterwarnings("ignore")

X = init(n_feature=10, m_sample=100)  # 样本数

X_name = ["X%s" % i for i in range(X.shape[1])]
for i, X_i in enumerate(X_name):
    locals()[X_i] = X[:, i]

# 添加关系
"""relation"""
'''
X2 = X0+X1
'''
"""noise"""
'''
X0 = add_noise(X0,ratio=0.01)
'''
"""定义函数"""  # 改变函数
y = X0 ** 2 + X2


X3 = add_noise(X0 / X2, ratio=0.2)
X4 = add_noise(X0 * X2, ratio=0.2)
X5 = add_noise(X1 + X2, ratio=0.2)
X6 = add_noise(X0 * X2, ratio=0.2)

X0 = add_noise(X0, ratio=0.35)
X2 = add_noise(X2, ratio=0.2)

"""重定义"""
X_all = [eval("X%s" % i) for i in range(X.shape[1])]
X_new = np.vstack(X_all).T

"""定义范围"""
feature_number = [2]

"""定义方法"""
estimator = method_pack(['KNR-set', 'SVR-set', "KR-set"], me="reg", gd=True)

feature_list = list(range(X_new.shape[1]))
slice_all = list(chain(*[list(combinations(feature_list, i)) for i in feature_number]))

pre_select = True

if pre_select:
    scores = []
    slice_all = []
    for estimatori in estimator:
        clf = Exhaustion(estimatori, n_select=feature_number, muti_grade=2, muti_index=None, must_index=None,
                         n_jobs=3)
        clf.fit(X_new, y)
        slice_all.extend(list(zip(*clf.score_ex[:10]))[0])
        scores.append(clf.score_ex[:10])
        print(clf.score_ex[0])
    slice_all = [tuple(_) for _ in slice_all]
    slice_all = list(set(slice_all))

# class MyTestCase(unittest.TestCase):
#     def setUp(self):
#         self.case = SUM(estimator, slice_all, estimator_n=None, n_jobs=3)
#         self.case.fit(X_new, y)
#
#     def test_something(self):
#         ss = self.case.cv_score_all(estimator_i=0)
#         # ss0 = self.case.cal_binary_distance_all(estimator_i=0)
#         # ss1 = self.case.cal_binary_distance_all(estimator_i=1)
#         # ss2 = self.case.cal_binary_distance_all(estimator_i=2)
#         # self.case.cluster_print(ss0)
#         # self.case.cluster_print(ss1)
#         # self.case.cluster_print(ss2)
#         # d = self.case.mean_max_method()
#         # c = self.case.distance_method()
#         # a = self.case.kk_distance_method()
#         # b = self.case.y_distance_method()
#         # e = self.case.pareto_method()
#         # f = self.case.y_distance_method()
#         # self.case.cal_group()
#         # self.case.cal_t_group()
#         #
#         # result = pd.DataFrame(np.array(ss))
#         # result = pd.DataFrame(np.array(result))
#         # result.to_csv('Result.csv')
#         self.assertIsInstance(ss, np.ndarray)

# if __name__ == '__main__':
#     a = unittest.main()

if __name__ == '__main__':
    reg = SUM(estimator, slice_all, estimator_n=None, n_jobs=3)
    reg.fit(X_new, y)

    # ss = reg.cv_score_all(estimator_i=0)
    # ss0 = reg.cal_binary_distance_all(estimator_i=0)
    # ss1 = reg.cal_binary_distance_all(estimator_i=1)
    # ss2 = reg.cal_binary_distance_all(estimator_i=2)
    # reg.cluster_print(ss0)
    # reg.cluster_print(ss1)
    # reg.cluster_print(ss2)

    # reg.cal_group()
    # reg.cal_t_group()
    #

    ma = reg.distance_method()
    mb = reg.y_distance_kk_method()
    mc = reg.mean_max_method()
    md = reg.y_distance_method()
    me = reg.distance_kk_method()
    mf = reg.pareto_method()

    # result = pd.DataFrame(np.array(a))
    # result = pd.DataFrame(np.array(result))
    # result.to_csv('Result.csv')
