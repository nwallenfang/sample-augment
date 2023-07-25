import os
import re

import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage.transform import resize

from sample_augment.models.generator import GC10_CLASSES, GC10_CLASSES_TEXT
from sample_augment.utils.path_utils import shared_dir
from sample_augment.utils.plot import prepare_latex_plot


def load_images(directory):
    images_dict = {}
    # class_ = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith(
                '.png'):
            img = mpimg.imread(os.path.join(directory, filename))
            # images.append(resize(img, (512, 512)))

            # Extract class name from filename (the img id part is optional)
            class_name = re.match(r"([a-z_]+)(?=_img|\.jpg$)", filename).group(1)
            images_dict[class_name] = resize(img, (512, 512))

    images = [images_dict[class_name] for class_name in GC10_CLASSES]
    return images


def make_class_overview():
    images = load_images(shared_dir / 'figures/class_representatives/')
    prepare_latex_plot()
    fig = plt.figure(figsize=(6, 8))  # create figure without subplots
    gs = GridSpec(4, 3)  # grid layout with 4 rows and 3 columns
    for i, img in enumerate(images):
        row, col = divmod(i, 3)  # calculate row and column index
        if i == len(images) - 1:  # if it is the last image
            ax = fig.add_subplot(gs[row, :])  # add subplot that spans the whole row
        else:
            ax = fig.add_subplot(gs[row, col])  # add normal subplot
        ax.imshow(img, cmap="gray")
        ax.axis('off')
        ax.set_title(f'{GC10_CLASSES_TEXT[i]}')
    # make remaining subplots invisible
    for j in range(len(images) + 1, 12):  # remaining subplot indices
        row, col = divmod(j, 3)
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(shared_dir / f'figures/class_representatives.pdf', bbox_inches="tight")


if __name__ == '__main__':
    make_class_overview()
