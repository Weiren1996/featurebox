#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/8/2 15:47
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

from setuptools import setup, find_packages

setup(
    name='featurebox',
    version='0.0.8',
    keywords=['features', "combination", "selection"],
    description='this is an box contains tools for machine learning.'
                'Some of code are non-originality, just copy for use. All the referenced code are marked,'
                'details can be shown in their sources',
    license='BSD 3-Clause',
    install_requires=['pandas', 'numpy', 'sympy', 'scipy', 'scikit-learn', 'joblib', 'matplotlib',
                      'networkx', 'seaborn', 'requests', 'tqdm', 'six', 'deap', 'scikit-image'],
    include_package_data=True,
    author='wangchangxin',
    author_email='986798607@qq.com',
    python_requires='>=3.6',
    url='https://github.com/boliqq07/featurebox',
    maintainer='wangchangxin',
    platforms=[
        "Windows",
        "Unix",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],

    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests", "tests", 'deprecated'], ),
    long_description='this is an box contains tools for machine learning.'
                     'Some of code are non-originality, just copy for use. All the referenced code are marked,'
                     'details can be shown in their sources',
)
