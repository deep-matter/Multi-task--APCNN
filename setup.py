#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='MultiTaskAttribute',
    version='0.1.0',
    description='Multi-Task Learning Framework for Binary Semantic Attribute Prediction',
    author='Youness El Brag',
    author_email='younsselbrag@gmail.com',
    url='https://github.com/deep-matter/Multi-task--APCNN',
    install_requires=[
        'torch==1.10.0',
        'torchvision==0.11.1',
        'numpy==1.21.2',
        'scikit-learn==0.24.2',
        'PyYAML==5.4.1',
    ],
    packages=find_packages(),
)