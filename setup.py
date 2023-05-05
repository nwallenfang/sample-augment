from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Thesis Project on Sampling Strategies for Synthetic Data Augmentation.',
    author='Nils Wallenfang',
    license='MIT',
    install_requires = [
        'kaggle',
        'dep2>=2.4.1',
        'click',
        'python-dotenv>=0.5.1',
    ]
)
