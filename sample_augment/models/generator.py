import os
import sys
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor

import sample_augment.models.stylegan2.legacy as legacy
from sample_augment.utils import log
from sample_augment.utils.path_utils import root_dir, shared_dir
from sample_augment.models.stylegan2.training.networks import Generator, Discriminator
from sample_augment.utils.log import SuppressSpecificPrint

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
    "Punching hole",
    "Welding line",
    "Crescent gap",
    "Water spot",
    "Oil spot",
    "Silk spot",
    "Inclusion",
    "Rolled pit",
    "Crease",
    "Waist folding"
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
    D: Discriminator

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
        log.debug(f'Generator device: {self.device}')

        warnings.filterwarnings("ignore", category=UserWarning,
                                module=".*upfirdn2d.*",
                                message=".*Failed to build CUDA kernels for upfirdn2d.*")

        with open(self.pkl_path, 'rb') as f:
            log.debug(f'pkl: {pkl_path}')
            data = legacy.load_network_pkl(f)
            self.G = data['G_ema'].to(self.device)
            # does there exist ema as well for D? not sure
            self.D = data['D'].to(self.device)
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
        if c.ndim == 1:
            c = c.unsqueeze(0)
        c = c.to(self.device)
        if seed:
            z = torch.from_numpy(np.random.RandomState(seed).randn(c.shape[0], self.G.z_dim)).to(self.device)
        else:
            z = torch.from_numpy(np.random.RandomState(self.seed).randn(c.shape[0], self.G.z_dim)).to(self.device)
        return self.G.mapping(z, c, truncation_psi=truncation_psi)

    def w_to_img(self, w: Union[torch.Tensor, np.ndarray]):
        if isinstance(w, np.ndarray):
            w = torch.from_numpy(w)
        if w.ndim == 2:
            w = w.unsqueeze(0)
        w = w.to(self.device)
        # assert w.ndim == 3, "expecting shape (n, G.num_ws, G.w_dim)"
        assert w.shape[1:] == (self.G.num_ws, self.G.w_dim), "expecting shape (n, G.num_ws, G.w_dim)"
        with SuppressSpecificPrint("Setting up PyTorch plugin \"upfirdn2d_plugin\"... Failed!"):
            imgs = self.G.synthesis(w)
        imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return imgs

    def img_into_discriminator(self, img, c):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.to(self.device)

        if isinstance(c, np.ndarray):
            c = torch.from_numpy(c)
        c = c.to(self.device)

        if img.ndim == 3:
            img = img.unsqueeze(0)
        if c.ndim == 1:
            c = c.unsqueeze(0)
        return self.D(img, c=c)

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
            c = torch.from_numpy(c).to(self.device)
        else:
            c = c.to(self.device)
        if c.ndim == 1:
            c = c.unsqueeze(0)
        assert c.ndim == 2, f"conditional vector c: expected shape (n, c_dim)"
        n: int = c.shape[0]

        # build input z vector (gaussian distribution)
        z = torch.from_numpy(np.random.RandomState(self.seed).randn(n, self.G.z_dim)).to(self.device)

        # with SuppressSpecificPrint("Setting up PyTorch plugin \"upfirdn2d_plugin\"... Failed!"):
        imgs = self.G(z, c, truncation_psi=truncation_psi, noise_mode=noise_mode)

        imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return imgs

    @staticmethod
    def load_from_name(name: str, seed: int = 100) -> "StyleGANGenerator":
        return StyleGANGenerator(pkl_path=root_dir / "TrainedStyleGAN" / f"{name}.pkl", seed=seed)


def generate_image_pool(num_imgs_per_class=200, generator_name: str = "wdataaug-028_012200", random_seed: int = 100):
    # now that we calc more imgs per class, we could do a class-wise diversity metric
    generator = StyleGANGenerator.load_from_name(generator_name, seed=random_seed)
    num_classes = generator.G.c_dim
    generated_root = shared_dir / "generated" / generator_name
    generated_root.mkdir(exist_ok=True)
    for class_index in range(num_classes):
        log.info(f'--- {GC10_CLASSES[class_index]} ---')
        # generate single label instances
        label = torch.zeros((num_imgs_per_class, num_classes))
        label[:, class_index] = 1.0

        class_imgs = generator.generate(c=label)
        (generated_root / GC10_CLASSES[class_index]).mkdir(exist_ok=True)
        for idx, img in enumerate(class_imgs):
            # Convert the tensor to a PIL Image
            img_pil = Image.fromarray(img.cpu().numpy(), 'RGB')
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
