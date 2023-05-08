from setuptools import find_packages, setup

setup(
    name='sampling-aug',
    packages=find_packages(include=['sampling_aug', 'sampling_aug.*']),
    # package_dir={'': 'src'},
    version='0.1.1',
    description='Thesis Project on Sampling Strategies for Synthetic Data Augmentation.',
    author='Nils Wallenfang',
    license='MIT',
    install_requires=[
        'kaggle',
        'click',
        'python-dotenv>=0.5.1',
        'xmltodict',
        'matplotlib',
        'numpy'
    ]
)
