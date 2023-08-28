import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances

from sample_augment.models.generator import StyleGANGenerator, GC10_CLASSES
from sample_augment.utils.path_utils import shared_dir


def load_projected_latents(projected_dir):
    X = []
    y = []
    indices = []
    class_names = []
    pattern = re.compile(r"^proj_([a-zA-Z_]+)_(\d+)\.npz$")

    # Go through each file in the dir and collect the latent vectors
    for file_name in os.listdir(projected_dir):
        if file_name.endswith(".npz"):
            # Extract class name and index from file name
            m = pattern.match(file_name)
            if m:
                class_name, index = m.groups()
                latent_vector = np.load(os.path.join(projected_dir, file_name))['w']
                X.append(latent_vector.reshape(-1))
                y.append(class_name)
                indices.append(int(index))  # keep track of index for future reference if needed
                if class_name not in class_names:
                    class_names.append(class_name)

    # Convert X, y, and index to arrays
    X = np.array(X)
    y = np.array(y)
    indices = np.array(indices)

    return X, y, indices


def style_mixing(latent_vector1, latent_vector2, crossover_layer):
    """
    Perform style mixing on two latent vectors.

    Args:
        latent_vector1 (np.array): The first latent vector.
        latent_vector2 (np.array): The second latent vector.
        crossover_layer (int): The layer at which to switch from latent_vector1 to latent_vector2.

    Returns:
        mixed_latent_vector (np.array): The resulting mixed latent vector.
    """
    mixed_latent_vector = np.copy(latent_vector1)
    mixed_latent_vector[crossover_layer:] = latent_vector2[crossover_layer:]
    return mixed_latent_vector


def test_random_w_lerp(generator: StyleGANGenerator):
    for i, class_name in enumerate(GC10_CLASSES):
        print(f'-- {class_name} --')
        c = torch.zeros((1, 10))
        c[0, i] = 1.0  # TODO could try label amplification here :), also analyze truncation psi
        start_latent = generator.c_to_w(c, seed=1).cpu().numpy()
        end_latent = generator.c_to_w(c, seed=2).cpu().numpy()
        plot_lerp_vs_mix(generator, start_latent, end_latent, fig_name=f'random_{class_name}.png')


def test_oversampling(projected_dir: Path, class_name: str, generator: StyleGANGenerator, start_index: int):
    """
    Test oversampling of latent vectors.

    Args:
        projected_dir (Path): The directory containing the projected latent vectors.
        class_name (str): The class of the projected latent to test.
        generator (StyleGANGenerator): The StyleGAN generator model.
        start_index (int): Index of the start image in the latent vector space.

    Returns:
        None
    """
    # Load the projected latents, class names, and indices
    X, y, indices = load_projected_latents(projected_dir)
    class_name_idx = GC10_CLASSES.index(class_name)
    c = torch.zeros(10)
    c[class_name_idx] = 1.0

    class_indices = np.where(y == class_name)[0]
    start_latent_index = min(start_index, len(class_indices) - 1)
    # TODO sometimes fewer than 20 instances - why? should print class counts
    start_latent = X[class_indices[start_latent_index]]

    # Exclude the start latent from the distance calculation
    other_indices = np.delete(class_indices, start_latent_index)
    other_latents = X[other_indices]

    distances = euclidean_distances(start_latent.reshape(1, -1), other_latents)
    # distances = euclidean_distances(start_latent.reshape(1, -1), X[class_indices])
    avg_distance = np.mean(distances)

    print(f"Average distance to class members: {avg_distance:.2f}")
    # TODO average distance to non-class members

    nearest_index = np.argmax(distances)  # changed to furthest distance :)
    print(f'Furthest distance: {distances[0, nearest_index]:.2f}')
    end_latent = other_latents[nearest_index]

    # linear interpolation to the nearest neighbor
    plot_lerp_vs_mix(generator, start_latent, end_latent, fig_name=f'mixing_vs_lerp_{class_name}_{start_index}.png')
 
def plot_lerp_vs_mix(generator, start_latent, end_latent, fig_name):
    steps = 4
    alpha_values = np.linspace(0, 1, steps+2)  # 4 in between, first and last are start and end
    interpolated_latents = np.array(
        [(1 - alpha) * start_latent + alpha * end_latent for alpha in alpha_values[1:-1]])

    # reshape start and nearest latent to original StyleGAN w latent shape
    start_latent = start_latent.reshape(14, 512)
    end_latent = end_latent.reshape(14, 512)

    # Generate images from the interpolated latents
    start_image = generator.w_to_img(start_latent).squeeze().cpu().numpy()
    end_image = generator.w_to_img(end_latent).squeeze().cpu().numpy()
    interpolated_images = generator.w_to_img(interpolated_latents.reshape((-1, 14, 512))).squeeze().cpu().numpy()

    crossover_layers = [1, 3, 5, 8, 12, 14] 
    mixed_images = []  # List to store the generated mixed images
    rev_mixed_images = []

    # Generate a mixed image for each crossover layer
    for crossover_layer in crossover_layers:
        mixed_latent = style_mixing(start_latent, end_latent, crossover_layer)
        mixed_image = generator.w_to_img(mixed_latent).squeeze().cpu().numpy()
        mixed_images.append(mixed_image)

    # now the other way around :)
    for crossover_layer in crossover_layers:
        mixed_latent = style_mixing(end_latent, start_latent, crossover_layer)
        mixed_image = generator.w_to_img(mixed_latent).squeeze().cpu().numpy()
        rev_mixed_images.append(mixed_image)

    # Plot start image, target image, interpolated images, and mixed image
    plot_images(start_image, end_image, interpolated_images, mixed_images, rev_mixed_images, crossover_layers,
                fig_name=fig_name)


def plot_images(start_image, end_image, interpolated_images, mixed_images, rev_mixed_images, crossover_layers, fig_name):
    """
    Plot images.

    Args:
        start_image (np.array): The start image.
        end_image (np.array): The end image.
        interpolated_images (list): The interpolated images (lerp).
        mixed_images (list): Images created from style mixing with different crossover layers.

    Returns:
        None
    """

    # Create a figure
    # fig, axes = plt.subplots(3, max(len(interpolated_images), len(mixed_images)) + 2, figsize=(20, 20))
    fig, axes = plt.subplots(3, 6, figsize=(20, 15))

    # Plot the start image
    axes[0, 0].imshow(start_image, cmap='gray')
    axes[0, 0].set_title('Start')

    # Plot the interpolated images
    for i, img in enumerate(interpolated_images):
        axes[0, i + 1].imshow(img, cmap='gray')
        axes[0, i + 1].set_title(f'Interpolated  {i + 1}')

    # Plot the end image
    axes[0, -1].imshow(end_image, cmap='gray')
    axes[0, -1].set_title('End')

    # Plot the mixed images
    for i, img in enumerate(reversed(mixed_images)):
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f'Mixed(Start, End, layer={crossover_layers[::-1][i]})')

    for i, img in enumerate(rev_mixed_images):
        axes[2, i].imshow(img, cmap='gray')
        axes[2, i].set_title(f'Mixed(End, Start, layer={crossover_layers[i]})')

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(shared_dir / 'figures/latents' / fig_name, bbox_inches='tight')


def main():
    
    generator = StyleGANGenerator.load_from_name('wdataaug-028_012200')
    test_random_w_lerp(generator)
    # for i, class_name in enumerate(GC10_CLASSES):
    #     print(f'-- {class_name} --')
    #     test_oversampling(projected_dir=shared_dir / 'projected',
    #                       class_name=class_name,
    #                       generator=generator,
    #                       start_index=2 + i
    #                       )


if __name__ == '__main__':
    main()
