import os
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt, image as mpimg
from torch import Tensor
from torchvision import transforms

from sample_augment.models.generator import StyleGANGenerator
from sample_augment.utils import log
from sample_augment.utils.path_utils import shared_dir


def interpolate(generator: StyleGANGenerator, start_latent: Tensor, end_latent: Tensor):
    alpha = 0.5
    interpolated_latent = (1 - alpha) * start_latent + alpha * end_latent
    # _interpolated_label = (1 - alpha) * start_label + alpha * end_label

    interpolated_img = generator.w_to_img(interpolated_latent).squeeze().cpu().permute(2, 0, 1)
    # interpolated_img = interpolated_img.permute(1, 2, 0)
    log.info(interpolated_img.shape)
    interpolated_img = transforms.ToPILImage()(interpolated_img)
    return interpolated_img


def plot_projection():
    img_dir = shared_dir / 'projected' / 'projected_plot'
    print(os.listdir(img_dir))

    generator = StyleGANGenerator.load_from_name('wdataaug-028_012200', seed=42)
    crescent_gap_paths = ['crescent_gap_168_proj.npz', 'crescent_gap_907_proj.npz']
    oil_spot_paths = ['oil_spot_40_proj.npz', 'oil_spot_886_proj.npz']
    crescent_gap = []
    oil_spot = []
    # load latents
    for c_latent in crescent_gap_paths:
        with np.load(str(img_dir / c_latent)) as data:
            target_latent = torch.tensor(data['w'])
            target_labels = torch.tensor(data['c'])
            crescent_gap.append(target_latent)
    for o_latent in oil_spot_paths:
        with np.load(str(img_dir / o_latent)) as data:
            target_latent = torch.tensor(data['w'])
            target_labels = torch.tensor(data['c'])
            oil_spot.append(target_latent)

    crescent_gap_interp = interpolate(generator, crescent_gap[0], crescent_gap[1])
    oil_spot_interp = interpolate(generator, oil_spot[0], oil_spot[1])

    crescent_gap_interp.save(str(img_dir / 'crescent_gap_interp.png'))
    oil_spot_interp.save(str(img_dir / 'oil_spot_interp.png'))


def plot_projection_grid(img_dir: Path):
    fig = plt.figure(figsize=(15, 10))

    # Define positions [left, bottom, width, height] for each type of image
    positions_real_proj = [
        [0.1, 0.55, 0.2, 0.4], [0.35, 0.55, 0.2, 0.4],
        [0.1, 0.05, 0.2, 0.4], [0.35, 0.05, 0.2, 0.4]
    ]
    positions_interp = [[0.7, 0.55, 0.2, 0.4], [0.7, 0.05, 0.2, 0.4]]

    files_real_proj = [
        'crescent_gap_168_target.png', 'crescent_gap_168_proj.png',
        'crescent_gap_907_target.png', 'crescent_gap_907_proj.png',
        'oil_spot_40_target.png', 'oil_spot_40_proj.png',
        'oil_spot_886_target.png', 'oil_spot_886_proj.png'
    ]

    files_interp = ['crescent_gap_interp.png', 'oil_spot_interp.png']

    # Plotting the Real and Projected images
    for pos, file in zip(positions_real_proj * 2, files_real_proj):
        ax = plt.axes(pos)
        img = mpimg.imread(img_dir / file)
        ax.imshow(img)
        ax.axis('off')

    # Plotting the Interpolated images
    for pos, file in zip(positions_interp, files_interp):
        ax = plt.axes(pos)
        img = mpimg.imread(img_dir / file)
        ax.imshow(img)
        ax.axis('off')

    # Add titles
    plt.figtext(0.2, 0.95, 'Real Instance', ha='center')
    plt.figtext(0.45, 0.95, 'Projected', ha='center')
    plt.figtext(0.75, 0.95, 'Interpolated', ha='center')

    plt.show()


if __name__ == '__main__':
    # plot_projection()
    plot_projection_grid(shared_dir / 'projected' / 'projected_plot')
