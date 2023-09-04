import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


def show_image_numpy(img: np.ndarray, title: str, text: str = None, save_path: str = None):
    _fig = plt.figure()
    _ax = plt.imshow(img)

    plt.title(title)
    if text:
        plt.figtext(0.5, 0.05, text, ha='center', fontsize=9)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
    
def show_image_tensor(img_tensor: torch.Tensor, title: str, text: str = None, save_path: str = None):
    img_numpy = img_tensor.cpu().permute(1, 2, 0).numpy()
    show_image_numpy(img_numpy, title, text, save_path)


text_width, text_height = 5.8476, 8.88242


def prepare_latex_plot(width=text_width, height=text_height):
    # set up matplotlib parameters to let LaTeX do the typesetting for a more unified look in the thesis
    # save matplotlib plots as pdf for this
    # matplotlib.rcParams['figure.figsize'] = (width, height)
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            # Adjust to your LaTex-Engine
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
            "axes.unicode_minus": False,
        }
    )
    # plt.rcParams.update({
    #     "pgf.texsystem": "pdflatex",
    #     "text.usetex": True,  #
    #     "font.family": "serif",
    #     "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    #     "font.sans-serif": [],
    #     "font.monospace": [],
    #     "axes.labelsize": 12,
    #     "font.size": 12,
    #     "legend.fontsize": 12,
    #     "xtick.labelsize": 12,
    #     "ytick.labelsize": 12,
    #     "pgf.preamble": r"\usepackage[utf8x]{inputenc} \usepackage[T1]{fontenc}"
    # })
    # plt.gcf().set_size_inches(text_width, text_height)
    # plt.tight_layout()


CREATE_PLOTS = True
