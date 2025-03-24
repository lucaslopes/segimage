from setuptools import setup, find_packages

setup(
    name='segimage',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'igraph',
        'scikit-image',
        'pycairo',
    ],
    entry_points={
        'console_scripts': [
            'img-segment=src.main:main',
        ],
    },
)
