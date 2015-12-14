#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

setup(
    name="scombine",
    version='0.1.0',
    author="Ben Johnson",
    author_email="benjamin.johnson@cfa.harvard.edu",
    packages=["scombine"],
    url="",
    license="LICENSE",
    description="simple SED generation from step-function SFHs",
    long_description=open("README.md").read() + "\n\n",
#                    + "Changelog\n"
#                    + "---------\n\n"
#                    + open("HISTORY.rst").read(),
    include_package_data=True,
    #install_requires=["numpy", "scipy >= 0.9", "astropy", "matplotlib", "scikit-learn"],
)
