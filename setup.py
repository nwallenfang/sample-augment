from setuptools import find_packages, setup

setup(
    name='thesis-nils',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    version='0.1.0',
    description='Thesis Project on Sampling Strategies for Synthetic Data Augmentation.',
    author='Nils Wallenfang',
    license='MIT',
    install_requires = [
        'kaggle',
        'click',
        'python-dotenv>=0.5.1',
        'xmltodict',
    ]
)
