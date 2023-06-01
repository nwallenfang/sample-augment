import matplotlib.pyplot as plt
import numpy as np

from sampling_aug.utils.paths import project_path


def show_image(img: np.ndarray, title: str, text: str, save_path: str = None):
    _fig = plt.figure()
    _ax = plt.imshow(img, cmap='gray')

    plt.title(title)
    plt.figtext(0.5, 0.05, text, ha='center', fontsize=9)

    if save_path:
        plt.savefig(project_path(save_path))
