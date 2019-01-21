"""setup.py: setuptools setup for minnetonka"""


from setuptools import setup, find_packages

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.txt')

setup(
	name="minnetonka",
	version=minnetonka.__version__,
	packages=['minnetonka', 'minnetonka.test'],
	description="A Python package for business modeling and simulation",
	long_description=long_description,
    author='David Bridgeland',
    author_email='dave@hangingsteel.com',
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.15.2',
        'scipy>=1.1.0'
        ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache License, Version 2.0 (Apache-2.0)'
        ]
	)