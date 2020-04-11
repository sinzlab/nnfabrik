#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="nnfabrik",
    version="0.0.0",
    description="Factory for Neural Networks",
    author="Konstantin Willeke, Edgar. Y. Walker",
    author_email="edgar.y.walker@mnf.uni-tuebingen.de",
    packages=find_packages(exclude=[]),
    install_requires=[],
)
