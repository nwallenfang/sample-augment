import matplotlib.pyplot as plt
import numpy as np
import torch

from sample_augment.models.generator import StyleGANGenerator
from sample_augment.utils import log
from sample_augment.utils.path_utils import shared_dir


def lerp(w1, w2, t):
    """
        Linear interpolation
        Expects 3-dimensional inputs of the shape (1, 14, 512).
        i.e. (1, G.num_ws, G.w_dim)
    @param w1:
    @param w2:
    @param t:
    @return:
    """
    t = np.full_like(w1, t)  # Create a numpy array with the same shape as v0 filled with the value t
    return (1 - t) * w1 + t * w2


def slerp(w1, w2, t):
    """
        Spherical interpolation.
        Motivation for slerp instead of lerp:
        "Remember that, unlike linear interpolation, spherical interpolation ensures that the interpolated vectors
         have the same norm (i.e., length) as the original vectors, which can be important in some cases when
         interpolating in high-dimensional spaces like latent spaces in GANs."
        expects tensor to be 2-D, i.e. (G.num_ws, G.w_dim)
    @param w1:
    @param w2:
    @param t:
    @return:
    """
    results = []
    """
        Slerp only "makes sense" when both vectors have the same norm. so they lie on some same hypersphere.
        This is not given for StyleGAN latent vectors and normalizing them (as is done right now) changes
        their semantics. For this reason, the idea of using spherical interpolation has been discarded.
    """
    w1 = w1 / np.linalg.norm(w1, axis=-1, keepdims=True)  # normalize v0
    w2 = w2 / np.linalg.norm(w2, axis=-1, keepdims=True)  # normalize v1
    # print(f"all_close: {np.allclose(w1, w2)}")
    for i in range(w1.shape[0]):
        # Slerp operation performed separately on each 512-D vector
        dotproduct = np.sum(w1[i] * w2[i])
        omega = np.arccos(np.clip(dotproduct, -1, 1))
        print(f"omega: {omega}")
        so = np.sin(omega)
        print(f"so: {so}")

        result = np.where(so != 0,
                          np.sin((1.0 - t) * omega) / so * w1[i] + np.sin(t * omega) / so * w2[i],
                          w1[i])
        results.append(result)

    return np.stack(results)


def main():
    generator = StyleGANGenerator.load_from_name('wdataaug-025_006200')

    # TODO: Your method to load and select pairs of representative w vectors for each class.
    # This is placeholder code and won't work as is.
    # pairs_of_w_vectors_for_each_class = [
    #     (np.load(str(shared_dir / f"projected/proj_proj_class_{class_idx}_image_1.npz"))['w'].squeeze(),
    #      np.load(str(shared_dir / f"projected/proj_proj_class_{class_idx}_image_2.npz"))['w'].squeeze())
    #     for class_idx in range(10)
    # ]
    pairs_of_w_vectors_for_each_class = []
    for class_idx in range(10):
        class_label = torch.zeros((1, 10))
        class_label[0][class_idx] = 1.0
        pairs_of_w_vectors_for_each_class.append((generator.c_to_w(c=class_label, seed=0).cpu().squeeze().numpy(),
                                                  generator.c_to_w(c=class_label, seed=1).cpu().squeeze().numpy()))
        # print(pairs_of_w_vectors_for_each_class[-1][0].shape)

    steps = 8  # Number of interpolation steps

    lerped_images_for_each_class = [
        [generator.w_to_img(lerp(w1, w2, i / steps)).squeeze().cpu().numpy() for i in range(steps + 1)]
        for w1, w2 in pairs_of_w_vectors_for_each_class
    ]

    fig, axes = plt.subplots(10, steps + 1, figsize=(20, 20))

    for class_idx, lerped_images in enumerate(lerped_images_for_each_class):
        for i, lerped_image in enumerate(lerped_images):
            axes[class_idx, i].imshow(lerped_image, cmap="gray")
            axes[class_idx, i].axis('off')

        axes[class_idx, 0].set_ylabel(f'Class {class_idx}')

    axes[0, 0].set_title('Linear interpolation')

    plt.tight_layout()
    plt.savefig(shared_dir / "figures" / "class_interpolation_experiment.pdf")
    log.info("Saved fig :)")


def lerp_vs_slerp():
    generator = StyleGANGenerator.load_from_name('wdataaug-025_006200')
    w1 = np.load(str(shared_dir / "projected/proj_proj_crescent_gap_98.npz"))['w'].squeeze()
    w2 = np.load(str(shared_dir / "projected/proj_proj_crescent_gap_427.npz"))['w'].squeeze()

    steps = 10  # Number of interpolation steps

    # Linear Interpolation
    lerped_ws = np.array([lerp(w1, w2, i / steps) for i in range(steps + 1)])
    lerped_images = [generator.w_to_img(w) for w in lerped_ws]

    # Spherical Interpolation
    slerped_ws = np.array([slerp(w1, w2, i / steps) for i in range(steps + 1)])
    slerped_images = [generator.w_to_img(w) for w in slerped_ws]

    print(len(lerped_images), len(slerped_images))

    fig, axes = plt.subplots(2, steps + 1, figsize=(20, 4))

    for i in range(steps + 1):
        lerped_image = lerped_images[i].squeeze().cpu().numpy()  # convert it to a numpy array (already transposed)
        axes[0, i].imshow(lerped_image, cmap="gray")
        axes[0, i].axis('off')

        slerped_image = slerped_images[i].squeeze().cpu().numpy()
        # print(f"slerped shape: {slerped_image.shape}")
        axes[1, i].imshow(slerped_image, cmap="gray")
        axes[1, i].axis('off')

    axes[0, 0].set_title('Linear interpolation')
    axes[1, 0].set_title('Spherical interpolation')

    plt.tight_layout()
    plt.savefig(shared_dir / "figures" / "interpolation_experiment.pdf")
    log.info("Saved fig :)")


if __name__ == '__main__':
    main()
    # lerp_vs_slerp()
