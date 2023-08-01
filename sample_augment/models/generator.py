import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor

import sample_augment.models.stylegan2.legacy as legacy
from sample_augment.utils import log
from sample_augment.utils.path_utils import root_dir, shared_dir
from sample_augment.models.stylegan2.training.networks import Generator

"""
    this file probably needs python <=3.8 and torch <=1.9, same as train_generator.
    As such it can't be included in the core framework with steps, etc.
    Instead we're relying on the shared directory and certain files just being well-formed.
"""

GC10_CLASSES = [
    "punching_hole",
    "welding_line",
    "crescent_gap",
    "water_spot",
    "oil_spot",
    "silk_spot",
    "inclusion",
    "rolled_pit",
    "crease",
    "waist_folding"
]
GC10_CLASSES_TEXT = [
    "punching hole",
    "welding line",
    "crescent gap",
    "water spot",
    "oil spot",
    "silk spot",
    "inclusion",
    "rolled pit",
    "crease",
    "waist folding"
]


class StyleGANGenerator:
    """
        where
            - z is the input latent space vector
            - w is intermediary synthesis latent space
            - c is the conditional (class-label) vector
        (this is the same terminology as the StyleGAN paper)
    """
    pkl_path: Path
    name: str
    device: torch.device
    seed: int
    G: Generator

    def __init__(self, pkl_path: Union[str, Path], seed: int = 100):
        if isinstance(pkl_path, str):
            self.pkl_path = Path(pkl_path)
        else:
            self.pkl_path = pkl_path

        self.name = pkl_path.name.split('.')[0].split('_')[0]
        self.out_dir = root_dir / "shared" / "generated" / self.name
        self.out_dir.mkdir(exist_ok=True)
        self.seed = seed

        os.makedirs(self.out_dir, exist_ok=True)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        log.info(f'Generator device: {self.device}')
        with open(self.pkl_path, 'rb') as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)

            if self.device.type == 'cpu':
                # see https://github.com/NVlabs/stylegan2-ada-pytorch/issues/105
                import functools
                self.G.forward = functools.partial(self.G.forward, force_fp32=True)

    def load_w_latent_vector(self, w_path: Path):
        ws = np.load(str(w_path))['w']
        ws = torch.tensor(ws, device=self.device)  # pylint: disable=not-callable
        assert ws.shape[1:] == (self.G.num_ws, self.G.w_dim)
        return ws

    def c_to_w(self, c: Tensor, truncation_psi: float = 1.0, seed: int = None) -> Tensor:
        c = c.to(self.device)
        if seed:
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).to(self.device)
        else:
            z = torch.from_numpy(np.random.RandomState(self.seed).randn(1, self.G.z_dim)).to(self.device)
        return self.G.mapping(z, c, truncation_psi=truncation_psi)

    def z_to_w(self, c: Tensor, z: Tensor) -> Tensor:
        """
        @param z:
        @param c: class label of the instance to be generated with the w
                    Shape: either (G.c_dim) or (n, G.c_dim).
                    If it's a stacked n-tensor with multiple labels, multiple images get generated
                    G.c_dim is the number_of_classes=10 for GC-10
        @return: uint8 image tensor with shape (C, H, W) = (3, 256, 256)
        """
        # TODO
        raise NotImplementedError()

    def z_to_img(self, c: Tensor, z: Tensor) -> Tensor:
        """
        @param c: class label of the instance to be generated with the w.
                    Shapes: either (num_classes) or (n, num_classes).
                    If it's a stacked n-tensor with multiple labels, multiple images get generated
        @param z: must match (n, G.z_dim) or (G.z_dim), depending on c
        @return: uint8 image tensor with shape (C, H, W) = (3, 256, 256)
        """
        # TODO
        raise NotImplementedError()

    def w_to_img(self, w: Union[torch.Tensor, np.ndarray]):
        if isinstance(w, np.ndarray):
            w = torch.from_numpy(w).to(self.device)
        if w.ndim == 2:
            w = w.unsqueeze(0)
        # assert w.ndim == 3, "expecting shape (n, G.num_ws, G.w_dim)"
        assert w.shape[1:] == (self.G.num_ws, self.G.w_dim), "expecting shape (n, G.num_ws, G.w_dim)"
        imgs = self.G.synthesis(w)
        imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return imgs
        # for idx, w in enumerate(w):
        #     img = self.G.synthesis(w.unsqueeze(0), noise_mode='const')
        #     img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        #     Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{self.out_dir}/proj{idx:02d}.png')

    # def generate_images(self,
    #                     truncation_psi: float = 1.0,
    #                     noise_mode='const',
    #                     save_to_outdir=False,
    #                     seeds=None,
    #                     class_idx=None):
    #
    #     # Synthesize the result of a W projection.
    #     assert self.G.c_dim != 0, "expected conditional network"
    #     # one-hot encoded class
    #     label = torch.zeros([1, self.G.c_dim], device=self.device)
    #     label[:, class_idx] = 1
    #
    #     imgs = np.empty((len(seeds), 256, 256, 3), dtype=np.uint8)
    #     # Generate images.
    #     for seed_idx, seed in enumerate(seeds):
    #         # print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
    #         z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).to(self.device)
    #         img = self.G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    #         img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    #         imgs[seed_idx] = img[0].cpu().numpy()
    #         if save_to_outdir:
    #             Image.fromarray(img[0].cpu().numpy(), 'RGB').save(
    #                 f'{self.out_dir}/{GC10_CLASSES[class_idx]}_{seed:04d}.png')
    #     return imgs

    def generate(self, c: Union[Tensor, np.ndarray],
                 truncation_psi: float = 1.0,
                 noise_mode='const') -> Tensor:
        """
        @param c: conditional label vector shape: (n, c_dim) with n being number of images to gen
        and c_dim number of classes.
        @param truncation_psi: param for tradeoff between diversity and fidelity, see StyleGAN paper
        @param noise_mode: idk
        @return: img stack tensor
        """
        if isinstance(c, np.ndarray):
            c = torch.from_numpy(c).unsqueeze(0).to(self.device)
        else:
            c = c.to(self.device)
        n: int = c.shape[0]
        # imgs = torch.empty((n, 256, 256, 3), dtype=torch.uint8)
        # build input z vector (gaussian distribution)
        z = torch.from_numpy(np.random.RandomState(self.seed).randn(n, self.G.z_dim)).to(self.device)

        # for i, seed in enumerate(list(range(n))):
        imgs = self.G(z, c, truncation_psi=truncation_psi, noise_mode=noise_mode)
        imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return imgs

    def generate_image_pool(self):
        pass

    @staticmethod
    def load_from_name(name: str) -> "StyleGANGenerator":
        return StyleGANGenerator(pkl_path=root_dir / "TrainedStyleGAN" / f"{name}.pkl")


def generate_image_pool(num_imgs_per_class=200, generator_name: str = "wdataaug-025_006200"):
    # now that we calc more imgs per class, we could do a class-wise diversity metric
    generator = StyleGANGenerator.load_from_name(generator_name)
    num_classes = StyleGANGenerator.G.c_dim
    generated_root = shared_dir / "generated" / generator_name
    for class_index in range(num_classes):
        print(f'--- {GC10_CLASSES[class_index]} ---')
        # generate single label instances
        label = torch.zeros((num_imgs_per_class, num_classes))
        label[class_index] = 1.0

        class_imgs = generator.generate(c=label)
        (generated_root / GC10_CLASSES[class_index]).mkdir(exist_ok=True)
        for idx, img in enumerate(class_imgs):
            # Convert the tensor to a PIL Image
            img_pil = Image.fromarray(img.numpy(), 'RGB')
            # Construct the file path
            file_path = generated_root / GC10_CLASSES[class_index] / f"{GC10_CLASSES[class_index]}_{idx}.png"
            # Save the image
            img_pil.save(file_path)


def main():
    if len(sys.argv) > 1:
        num_imgs_per_class = int(sys.argv[1])
    else:
        num_imgs_per_class = 200

    generate_image_pool(num_imgs_per_class)


if __name__ == '__main__':
    main()
