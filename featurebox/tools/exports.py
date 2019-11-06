#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/29 19:48
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause


"""
# Just a copy from xenonpy
"""

import os
from os import remove

import joblib
import numpy as np
import pandas as pd


class Store(object):
    def __init__(self, path=None, filename="filename", prefix: str = None):
        r"""
        store file in path
        :param filename: str,universal filename
        :param path: None, ..
        \data_cluster, or F:data_cluster\data1
        """

        if not prefix:
            prefix = ""
        self._prefix = prefix

        if path is None:
            path = os.getcwd()
        if os.path.exists(path):
            os.chdir(path)
        else:
            os.makedirs(path)
            os.chdir(path)

        self._path = path
        self._filename = ""
        self.default_filename = filename
        self._file_list = []

    def __repr__(self):
        return "store to ({}) with {} file".format(self._path, len(self.stored_file))

    __str__ = __repr__

    def _check_name(self, suffix="csv", file_new_name="filename", model="w"):

        self._filename = file_new_name or self.default_filename

        if os.path.isfile('{}{}.{}'.format(self._prefix, self._filename, suffix)) and model == "w":
            shu1 = 1
            while os.path.isfile('{}{}({}).{}'.format(self._prefix, self._filename, shu1, suffix)):
                shu1 += 1
            self._filename = '{}{}({}).{}'.format(self._prefix, self._filename, shu1, suffix)
        else:
            self._filename = '{}{}.{}'.format(self._prefix, self._filename, suffix)

        if self._filename in self._file_list:
            self._file_list.remove(self._filename)
        self._file_list.append(self._filename)

    def to_csv(self, data, file_new_name=None, model="w"):

        self._check_name("csv", file_new_name, model=model)
        if isinstance(data, (dict, list)):
            data = pd.DataFrame.from_dict(data)
        elif isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        if isinstance(data, pd.DataFrame):
            data.to_csv(path_or_buf="%s" % self._filename, sep=",", na_rep='', float_format=None,
                        columns=None, header=True, index=True, index_label=None,
                        mode=model, encoding=None, )

        else:
            raise TypeError("Not support data_cluster type:%s for csv" % type(data))

    def to_txt(self, data, file_new_name=None, model="w"):
        self._check_name("txt", file_new_name, model=model)
        document = open(self._filename, model)
        document.write(str(data))
        document.close()

    def to_pkl_pd(self, data, file_new_name=None):
        self._check_name("pkl.pd", file_new_name)
        pd.to_pickle(data, self._filename)

    def to_pkl_sk(self, data, file_new_name=None):
        self._check_name("pkl.sk", file_new_name)
        joblib.dump(data, self._filename)

    def remove(self, index_or_name=None):
        """
        remove the indexed file
        :param index_or_name: index or name,default=-1

        """
        if isinstance(index_or_name, str):
            name = index_or_name
            index = -1
        elif isinstance(index_or_name, int):
            name = None
            index = index_or_name
        else:
            name = None
            index = -1
        if not name:
            try:
                files = self._file_list[index]
            except IndexError:
                raise IndexError("No flie or wrong index to remove")
            else:
                if not isinstance(files, list):
                    remove(str(files))
                else:
                    for f in files:
                        remove(str(f))
                del self._file_list[index]

        elif name in self._file_list:
            if not isinstance(name, list):
                remove(str(name))
                self._file_list.remove(name)
            else:
                for f in name:
                    remove(str(f))
                    self._file_list.remove(f)
                self._file_list.remove([])

        else:
            raise NameError("No flie named %s" % index_or_name)

    def withdraw(self):
        """
        delet all stored_file file
        """
        files = self._file_list
        for f in files:
            remove(str(f))
        self._file_list = []
        self._filename = ""

    @property
    def stored_file(self):
        [print(i) for i in self._file_list]
        return self._file_list


if __name__ == "__main__":
    a = np.array([[1, 2], [3, 4]])
    st = Store(r'C:\Users\Administrator\Desktop')
    st.to_txt(a, file_new_name="filename", model="a+")
