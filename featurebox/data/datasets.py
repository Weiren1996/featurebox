# -*- coding: utf-8 -*-

# @Time    : 2019/11/6 16:20
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

import numpy as np
from sklearn.utils._random import check_random_state

'''
    """please copy the code to test dont use by import"""

    X = init(n_feature=10, m_sample=100)  # 样本数

    X_name = ["X%s" % i for i in range(X.shape[1])]
    for i, X_i in enumerate(X_name):
        locals()[X_i] = X[:, i]

    # 添加关系
    """relation"""

    X2 = X0+X1

    """noise"""

    X0 = add_noise(X0,ratio=0.01)


    X0 = add_noise(X0, ratio=0.2)
    X3 = add_noise(X0 / X2,ratio=0.2)
    X4 = add_noise(X0 * X2,ratio=0.2)

    """重定义"""
    X_all = [eval("X%s" % i) for i in range(X.shape[1])]
    X_new = np.vstack(X_all).T

    """定义函数"""  # 改变函数
    y = X0 ** 3 + X2
    return X, y
'''


def init(m_sample=100, n_feature=10):
    n = n_feature
    m = m_sample
    mean = [1] * n
    cov = np.zeros((n, n))
    for i in range(cov.shape[1]):
        cov[i, i] = 1
    rdd = check_random_state(1)
    X = rdd.multivariate_normal(mean, cov, m)
    return X


def add_noise(s, ratio):
    print(s.shape)
    rdd = check_random_state(1)
    return s + rdd.random_sample(s.shape) * np.max(s) * ratio


from sklearn.utils import Bunch


class CAMData(object):
    def __init__(self, samples, shape_x, random_state=None, method="poisson", inter=True, cut_zero=True,
                 re_range=None,
                 question="reg", rank=True):
        """

        Parameters
        ----------
        samples:int
        shape_x:int or tuple of int
        random_state:None or int
        method:"poission" or None
        inter:bool
            integer or not.
        cut_zero:
            non-negative or not
        re_range:None or tuple of int, default is (0,1)
            range
        question:"reg" or "clf"
        rank:bool
            rank values for y, just for the "reg"
        """
        if isinstance(shape_x, int):
            self.shape_x = (shape_x,)
        else:
            self.shape_x = shape_x

        self.rdd = check_random_state(random_state)
        self.random_state = random_state
        self.samples = samples
        self.method = method
        self.inter = inter
        self.cut_zero = cut_zero
        self.re_range = re_range
        self.question = question
        self.rank = rank

    def _gen_X(self):
        rdd = self.rdd
        method = self.method
        inter = self.inter
        cut_zero = self.cut_zero
        re_range = self.re_range

        if method == "poission":

            data = rdd.poisson(lam=1.0, size=(self.samples, *self.shape_x))
        else:
            data = rdd.random_sample(size=(self.samples, *self.shape_x))

        if re_range is None:
            pass
        elif re_range == "zeroone":
            a, b = 0, 1
            data = (b - a) * data + a
        else:
            a, b = re_range[0], re_range[1]
            data = (b - a) * data + a
        if inter:
            data = np.round(data)
        if cut_zero:
            data[np.where(data < 0)] = 0
        return data

    def gen_x(self, featurize_x=True, self_func=None):
        """

        Parameters
        ----------
        featurize_x: bool or "total" or "each"
            deal with x.
            if True or "each", the self_func should be a func to deal with each xi sample.
            input parameters are (xi,i,random_state)
            such as:

            def self_func(xi,i,random_state):
                return xi[0,0,0]+=i+random_state.randint(7,11,1), "random add"

            if "total", the self_func should be a func to deal with x data.
            input parameters are (self,x)
            such as:

            def featurized_x(self, x)
                return x*=2, "x*2"

            if False, pass
        self_func : callable object,or None,or str, default is None
            1.None: return xi and ""
            2.str:"cp_channel_to3","de_2Dsite_i"
            3.the function to deal to xi, return xi_new and feature messagei
            4.or the function to deal to x, return x_new and feature message

        """
        if self_func is None:
            self._featurized_xi = self._no_featurize_xi
        elif self_func is "cp_channel_to3":
            self._featurized_xi = self.cp_channel_to3
        elif self_func is "de_2Dsite_i":
            self._featurized_xi = self.de_2Dsite_i
        elif self_func is "de_2Dsite":
            self._featurized_xi = self.de_2Dsite
        else:
            raise TypeError(
                "self_func is callable object,or None,or 'cp_pipeline_to3','de_2Dsite','de_2Dsite_i', default is None")

        x = self._gen_X()

        if not featurize_x:
            self.feature = None
        elif featurize_x is True or featurize_x is "each":
            x, self.feature = self.featurized_x(x)
        elif featurize_x is "total":
            self.featurized_x = self._featurized_xi
            x, self.feature = self.featurized_x(x)
        else:
            raise TypeError('featurize_x is bool or "total" or "each"')

        return x

    @staticmethod
    def de_2Dsite_i(xi, i, random_state):
        """
        one instance just for 2d data
        a manipulation on xi and get new xi and feature message"""
        c = random_state.randint(7, 11, 1) / 10
        # c = random_state.random(1)
        b_a = i / c
        a = random_state.randint(0, b_a + 1, size=1)
        b = b_a + a

        ainx, ainy = random_state.randint(0, 10, size=2)
        ain = ainx, ainy
        bin = ainx + 5, ainy + 7
        cin = ainx + 2, ainy + 5
        xi[ain] = a
        xi[bin] = b
        xi[cin] = c
        featurei = ((ain, bin, cin), (a, b, c))
        # print(xi[cin]*(xi[bin]-xi[ain]))

        return xi, featurei

    def de_2Dsite(self, x):
        """resort the x"""

        def site(xi, i, rd):
            a, b, c = rd.random_sample(3)
            ainx, ainy = rd.randint(2, 10, size=2)
            ain = ainx, ainy
            ratio = rd.randint(10, 15, size=1) / 10
            bin = int(ainx + 13 * ratio), int(ainy + 1 * ratio)
            cin = int(ainx + 7 * ratio), int(ainy + 14 * ratio)
            xi[ain] = a
            xi[bin] = b
            xi[cin] = c
            featurei = ((ain, bin, cin), (b - a) * c)
            return xi, featurei

        if self._featurized_xi is None:
            return x, ""
        else:
            rd = self.rdd
            s1, *ss = x.shape
            x_new = []
            feature = []
            rank = []
            for i in range(s1):
                xi = x[i]
                xi, featurei = site(xi, i, rd)
                xi = np.expand_dims(xi, axis=0)
                x_new.append(xi)
                feature.append(featurei)
                rank.append(featurei[1])
            index = np.argsort(rank)
            x_new = np.array(x_new)[index]
            feature = np.array(feature)[index]
            x_new = np.concatenate(x_new, axis=0)
            return x_new, feature

    @staticmethod
    def cp_channel_to3(x):
        x = np.expand_dims(x, axis=-1)
        x = np.repeat(x, 3, axis=-1)

        return x, "dim (sample, ... , 3)"

    @staticmethod
    def cp_channel_to1(x):
        x = np.expand_dims(x, axis=-1)
        x = np.repeat(x, 1, axis=-1)

        return x, "dim (sample, ... , 1)"

    @staticmethod
    def _no_featurize_xi(xi, *kwargs):

        raise NotImplementedError("no self function")

    def featurized_x(self, x):
        if self._featurized_xi is None:
            return x, ""
        else:
            rd = self.rdd
            s1, *ss = x.shape
            x_new = []
            feature = []
            for i in range(s1):
                xi = x[i]
                xi, featurei = self._featurized_xi(xi, i, rd)
                xi = np.expand_dims(xi, axis=0)
                x_new.append(xi)
                feature.append(featurei)

            x_new = np.concatenate(x_new, axis=0)
            return x_new, feature

    def gen_y(self):
        question = self.question
        rank = self.rank
        shape_y = self.samples
        rdd = self.rdd
        if question is "reg":
            if rank:
                target = np.array(list(range(shape_y))) / shape_y * (1 + rdd.random_sample(shape_y) * 0.1)
                target = np.reshape(target, (shape_y, 1))
            else:
                target = rdd.random_sample(size=(shape_y, 1))
        else:
            if rank:
                target1 = np.zeros((int(shape_y / 2), 1))
                target2 = np.ones((shape_y - int(shape_y / 2), 1))
                target = np.concatenate((target1, target2), axis=0)
            else:
                target = rdd.randint(0, 2, size=(shape_y, 1))
        return target

    def gen_data(self, featurize_x=False, self_func=None):
        x = self.gen_x(featurize_x=featurize_x, self_func=self_func)
        y = self.gen_y()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        return Bunch(x=x, y=y, feature=self.feature)


if __name__ == "__main__":
    sample = 5000

    camdata = CAMData(15, 15, random_state=1, question="reg", inter=False, re_range=(-0.5, 1))
    data = camdata.gen_data(featurize_x=False)
    x = data.x
    y = data.y
    x = x.astype(np.float32)

    # t_x = trans_to_tensor(x)
