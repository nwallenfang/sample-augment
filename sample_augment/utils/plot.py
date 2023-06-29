import matplotlib.pyplot as plt
import numpy as np

from sample_augment.utils.path_utils import project_path


def show_image(img: np.ndarray, title: str, text: str, save_path: str = None):
    _fig = plt.figure()
    _ax = plt.imshow(img, cmap='gray')

    plt.title(title)
    plt.figtext(0.5, 0.05, text, ha='center', fontsize=9)

    if save_path:
        plt.savefig(project_path(save_path))


def prepare_latex_plot():
    # set up matplotlib parameters to let LaTeX do the typesetting for a more unified look in the thesis
    # save matplotlib plots as pdf for this
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,  #
        "font.family": "serif",
        "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "pgf.preamble": r"\usepackage[utf8x]{inputenc} \usepackage[T1]{fontenc}"
    })
