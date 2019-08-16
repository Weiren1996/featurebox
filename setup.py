#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/8/2 15:47
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

from setuptools import setup, find_packages

setup(
    name = 'featurebox',
    version = '0.0.4',
    # keywords = ('chinesename',),
    description = 'get a chinesename by random',
    license = 'MIT License',
    install_requires = ['pymatgen', 'pandas'],
    packages = ['chinesename'],  # 要打包的项目文件夹
    include_package_data=True,   # 自动打包文件夹内所有数据
    author = 'pengshiyu',
    author_email = 'pengshiyuyx@gmail.com',
    url = 'https://github.com/mouday/chinesename',
    # packages = find_packages(include=("*"),),
)
