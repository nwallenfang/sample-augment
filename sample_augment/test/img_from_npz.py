from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from sample_augment.models.generator import StyleGANGenerator
from sample_augment.utils.path_utils import shared_dir


def main():
    pass


if __name__ == '__main__':
    npz_path = shared_dir / 'projected' / 'latent_wplus_wavg_punching_hole_660.npz'
    generator = StyleGANGenerator.load_from_name('wdataaug-028_012200')
    w = np.load(str(npz_path))['w']
    img = generator.w_to_img(w)
    img = img.cpu().squeeze().numpy()
    print("frust")
    print(img.shape)
    Image.fromarray(img).save(str(shared_dir / 'projected' / 'hurz.png'))
    plt.imshow(img)
    plt.show()
