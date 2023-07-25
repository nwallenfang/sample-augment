import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

import sample_augment.models.stylegan2.legacy as legacy
from sample_augment.utils.path_utils import root_dir

"""
    this file probably needs python 3.7 and torch 1.7 same as train_generator.
    As such it can't be included in the core framework with steps, etc.
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
    pkl_path: Path
    """where generated images will get saved"""
    out_dir: Path
    name: str
    device: torch.device

    def __init__(self, pkl_path: Union[str, Path], out_dir=None):
        if isinstance(pkl_path, str):
            self.pkl_path = Path(pkl_path)
        else:
            self.pkl_path = pkl_path

        self.name = pkl_path.name.split('.')[0].split('_')[0]

        if out_dir:
            self.out_dir = out_dir
        else:
            self.out_dir = root_dir / "shared" / "generated" / self.name
        self.out_dir.mkdir(exist_ok=True)

        os.makedirs(self.out_dir, exist_ok=True)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        with open(self.pkl_path, 'rb') as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)

            if self.device.type == 'cpu':
                # see https://github.com/NVlabs/stylegan2-ada-pytorch/issues/105
                import functools
                self.G.forward = functools.partial(self.G.forward, force_fp32=True)

    # custom generate method without the click context for programmatic access
    def generate_images(self,
                        truncation_psi: float = 1.0,
                        noise_mode='const',
                        save_to_outdir=False,
                        seeds=None,
                        class_idx=None,
                        projected_w=None):

        # Synthesize the result of a W projection.
        if projected_w is not None:
            if seeds is not None:
                print('warn: --seeds is ignored when using --projected-w')
            print(f'Generating images from projected W "{projected_w}"')
            ws = np.load(projected_w)['w']
            ws = torch.tensor(ws, device=self.device)  # pylint: disable=not-callable
            assert ws.shape[1:] == (self.G.num_ws, self.G.w_dim)
            for idx, w in enumerate(ws):
                # TODO check out the synthesis method
                img = self.G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{self.out_dir}/proj{idx:02d}.png')
            return

        if seeds is None:
            print('--seeds option is required when not using --projected-w')

        assert self.G.c_dim != 0, "expected conditional network"
        # one-hot encoded class
        label = torch.zeros([1, self.G.c_dim], device=self.device)
        label[:, class_idx] = 1

        imgs = np.empty((len(seeds), 256, 256, 3), dtype=np.uint8)
        # Generate images.
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).to(self.device)
            img = self.G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs[seed_idx] = img[0].cpu().numpy()
            if save_to_outdir:
                Image.fromarray(img[0].cpu().numpy(), 'RGB').save(
                    f'{self.out_dir}/{GC10_CLASSES[class_idx]}_{seed:04d}.png')
        return imgs


if __name__ == '__main__':
    num_classes = 10
    if len(sys.argv) > 1:
        num_imgs_per_class = int(sys.argv[1])
    else:
        num_imgs_per_class = 200

    # now that we calc more imgs per class, we could do a class-wise diversity metric
    generator = StyleGANGenerator(pkl_path=root_dir / 'TrainedStyleGAN/apa-020_004400.pkl')

    for class_index in range(num_classes):
        print(f'--- {GC10_CLASSES[class_index]} ---')
        generated_imgs = generator.generate_images(class_idx=class_index,
                                                   seeds=list(range(num_imgs_per_class)),
                                                   save_to_outdir=True)
