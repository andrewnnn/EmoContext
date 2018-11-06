#!/usr/bin/env python

from setuptools import setup, find_packages
project_name="d2d-template"
project_version = '0.0.1'
description = "Template project"
setup(
    name=project_name,
    version=project_version,
    packages=find_packages(),
    url='https://www.d2dcrc.com.au',
    license='COMMERCIAL',
    author='d2dcrc',
    author_email='info@d2dcrc.com.au',
    description=description,
    install_requires=[
        "numpy",
        "scikit-learn",
        "python-dateutil"
    ],
    #package_data={'d2d.event_size': ['resources/*']},
)
