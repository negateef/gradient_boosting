from setuptools import setup
import sys

setup(
    name='GradientBoostingRegressor',
    version='1.0',
    packages=['gradient_boosting'],
    install_requires=[
        'Click',
        'numpy',
        'pandas',
        'sklearn',
        'scipy'
    ],
    entry_points={'console_scripts' : [
        'gradient_boosting=gradient_boosting.gradientBoostingCLI:main'
    ]},
    author='Mykhailo Babenko',
    author_email='miwaqq@gmail.com',
    licence=open('LICENSE').read()
)