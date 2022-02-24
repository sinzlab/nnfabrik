#!/usr/bin/env python
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

setup(
    name="nnfabrik",
    version="0.1.0",
    description="A generalized model fitting pipeline",
    author="Sinz Lab",
    author_email="software@sinzlab.net",
    packages=find_packages(exclude=[]),
    install_requires=[
        "sphinx",
        "pytorch_sphinx_theme",
        "recommonmark",
        "ax_platform",
    ],
)
