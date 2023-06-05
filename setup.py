from setuptools import setup

setup(
    name='boxlib',
    version='0.1.0',
    author='István Sárándi',
    author_email='istvan.sarandi@uni-tuebingen.de',
    packages=['boxlib'],
    license='LICENSE',
    description='Utilities for bounding box manipulation',
    python_requires='>=3.6',
    install_requires=['numpy']
)
