#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
setup(
    name="cmod",
    version="0.1",
    packages=find_packages('cmod'),
    include_package_data=True,
    # scripts=['cmod.py', 'cmod_gpu.py', 'cmod_vect.py'],

    # metadata for upload to PyPI
    author="Alexander Myasoedov",
    author_email="mag@rshu.ru",
    description="Calculates Normalized Radar Cross Section using CMOD4/5 model",
    keywords="cmod, sar, nrcs",
    url="http://github.com/SOLab/cmod",   # project home page
)
