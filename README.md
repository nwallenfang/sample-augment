# Sample-Augment: Tool for Synthetic Data Augmentation

This is a Master's Thesis Project exploring Sampling Strategies for Synthetic Data Augmentation utilizing
Generative Models, such as Generative Adversarial Networks (GANs). The objective is to augment a dataset
synthetically to improve performance on downstream tasks such as image classifcation.

As of June 2023, the foundation of the project is established, including the basic
pipeline and modules for data processing and model training. The next steps involve training a Generator
in the form of StyleGAN2-ADA and implementing different sampling strategies based on it.

## Project Structure

The main Python package for this project is located in [`sample_augment/`](sample_augment). It contains two
main components:

- [`core/`](sample_augment/core): A general framework for constructing reproducible and
  serializable pipelines tailored to Machine Learning needs. Its main features are an interface for 
  pipeline steps with dependency resolution, centralized Config and intermediary Data management. 

- The other modules are specific to the thesis:
    - [`data/`](sample_augment/data): Scripts for data processing steps, such as 
      preprocessing, train-test split. It also contains an adaptor for the GC10-DET dataset which gets 
      covered in this thesis.
    - [`models/`](sample_augment/models): Scripts for training and evaluating the 
      classifier and generator models.

## Installation

First, initialize an Anaconda environment. You can use one of the `environment.yml` files to replicate a
working environment.

- `environment_cpu.yml`
    - Python 3.10 Environment for local development without CUDA installed.
    - Can use this for anything but training/evaluating models
    - This environment was tested on a local Ubuntu 22 installation.
- `environment_gpu.yml`
    - Python 3.10 environment with CUDA installed
    - This environment was used for running experiments (Classifier training, etc.) on the IPT DS
      machine `ipt-d-0432`
    - This can NOT run StyleGAN
    - TODO provide this env file
- `environment_stylegan.yml`
    - StyleGAN depends on a specific version of PyTorch and on Python 3.7
    - not every method in `sampling_aug` is compatible with this configuration, so only use this when
      training/evaluating StyleGAN
    - TODO provide this env file
      Enter this directory and call `pip install -e .`.
      Pytorch installation instructions: TODO

## License
The project is licensed under [MIT](LICENSE).

Note that this repository contains a submodule pointing to a fork of [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch), which is
released under a license for non-commercial use.
