#
# analytic_computation setuptools script
#
from setuptools import setup, find_packages


def get_version():
    """
    Get version number from the ebola_model module.
    """
    import os
    import sys

    sys.path.append(os.path.abspath('ebola_model'))
    from version_info import VERSION as version
    sys.path.pop()

    return version


def get_requirements():
    requirements = []
    with open("requirements.txt", "r") as file:
        for line in file:
            requirements.append(line)
    return requirements


setup(
    # Module name
    name='ebola_model',

    # Version
    version=get_version(),

    description='A base package for calculating the major outbreak probability',

    maintainer='Abbie Evans',

    maintainer_email='abbie.evans@keble.ox.ac.uk',

    url='https://github.com/abbie-evans/ebola-outbreak-risks',

    # Packages to include
    packages=find_packages(include=('ebola_model', 'ebola_model.*')),

    # List of dependencies
    install_requires=get_requirements(),

    extras_require={
        'docs': [
            'sphinx>=1.5, !=1.7.3',
        ],
        'dev': [
            'flake8>=3',
            'pytest',
            'pytest-cov',
        ],
    },
)