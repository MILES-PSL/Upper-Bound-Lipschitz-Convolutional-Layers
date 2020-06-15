from os.path import dirname, join
from setuptools import setup, find_packages


setup(
    name='LipschitzBound',
    version=0.1,
    description='Compute the max singular value of convolution',
    author='Alexandre Araujo',
    maintainer='Alexandre Araujo',
    maintainer_email='alexandre.araujo@dauphine.eu',
    license='MIT',
    packages=find_packages(exclude=('docs', 'tests')),
    zip_safe=False,
    install_requires=[
    ],
)
