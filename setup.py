from setuptools import find_packages, setup

setup(
    name='thesis_nils',
    packages=['thesis_nils'],
    package_dir={'': 'src'},
    version='0.1.1',
    description='Thesis Project on Sampling Strategies for Synthetic Data Augmentation.',
    author='Nils Wallenfang',
    license='MIT',
    install_requires=[
        'kaggle',
        'click',
        'python-dotenv>=0.5.1',
        'xmltodict',
    ]
)
