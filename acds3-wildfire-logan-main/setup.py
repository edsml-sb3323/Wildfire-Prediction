from setuptools import setup, find_packages

setup(
    name='WildfireLogan',
    version='0.1.0',
    packages=find_packages(),
    license="MIT",
    install_requires=[
        'numpy',
        'torch',
        'scikit-learn',
        'matplotlib',
        'livelossplot'
    ],
    url="https://github.com/ese-msc-2023/acds3-wildfire-logan.git",
    python_requires='>=3.6',
)
