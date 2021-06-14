from setuptools import setup
from ctdataset import __version__

setup(
    name='ctdataset',
    version=__version__,
    packages=[
        "ctdataset",
        "ctdataset.dataset",
    ],
    include_package_data=True,
    url='https://github.com/JeanMaximilienCadic/ctdataset',
    license='MIT',
    author='Jean Maximilien Cadic',
    python_requires='>=3.6',
    install_requires=[d.rsplit()[0] for d in open("requirements.txt").readlines()],
    author_email='support@cadic.jp',
    description='ctdataset',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
    ]
)

