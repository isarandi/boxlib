from setuptools import setup
import os

try:
    dependencies_managed_by_conda = os.environ['DEPENDENCIES_MANAGED_BY_CONDA'] == '1'
except KeyError:
    dependencies_managed_by_conda = False

setup(
    name='boxlib',
    version='0.1.0',
    author='István Sárándi',
    author_email='istvan.sarandi@uni-tuebingen.de',
    packages=['boxlib'],
    license='LICENSE',
    description='Utilities for bounding box manipulation',
    python_requires='>=3.6',
    install_requires=[] if dependencies_managed_by_conda else ['numpy']
)
