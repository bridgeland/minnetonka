"""setup.py: setuptools setup for minnetonka"""

import os
from setuptools import setup 


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
	name="minnetonka",
	version='0.0.1',
    py_modules=['minnetonka', 'test'],
	description="A Python package for business modeling and simulation",
	long_description=read('README.rst'),
    long_description_content_type='text/x-rst',
    author='David Bridgeland',
    author_email='dave@hangingsteel.com',
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.15.2',
        'scipy>=1.1.0'
        ],
    url='https://github.com/bridgeland/minnetonka',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License'
        ]
	)