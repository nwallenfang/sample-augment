Thesis Nils
==============================

Thesis Project on Sampling Strategies for Synthetic Data Augmentation.

## Installation
First, initialize an Anaconda environment. You can use one of the `environment.yml` files to replicate a working environment.
- `environment_cpu.yml`
  - Python 3.10 Environment for local development without CUDA installed.
  - Can use this for anything but training/evaluating models
  - This environment was tested on a local Ubuntu 22 installation.
- `environment_gpu.yml`
  - Python 3.10 environment with CUDA installed
  - This environment was used for running experiments (Classifier training, etc.) on the IPT DS machine `ipt-d-0432`
  - This can NOT run StyleGAN
  - TODO
- `environment_stylegan.yml`
  - StyleGAN depends on a specific version of PyTorch and on Python 3.7
  - not every method in `sampling_aug` is compatible with this configuration, so only use this when training/evaluating StyleGAN
  - TODO
Enter this directory and call `pip install -e .`.
Pytorch installation instructions: TODO

## Project Organization
The main Python package is located in [sampling_aug/](sampling_aug). All datasets that are generated in `sampling_aug` get saved under [data/](data).
Classifier checkpoints do get saved in [models/](models), though StyleGAN checkpoints might get saved somewhere else depending on their specific output directory.


## License
This repository contains a fork of [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch), which is released under a license for non-commercial use.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
