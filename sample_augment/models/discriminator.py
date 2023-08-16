import warnings
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import Tensor

import sample_augment.models.stylegan2.legacy as legacy
from sample_augment.models.stylegan2.training.networks import Discriminator
from sample_augment.utils import log
from sample_augment.utils.path_utils import root_dir


class StyleGANDiscriminator:
    pkl_path: Path
    name: str
    device: torch.device
    seed: int
    D: Discriminator

    def __init__(self, pkl_path: Union[str, Path], seed: int = 100):
        if isinstance(pkl_path, str):
            self.pkl_path = Path(pkl_path)
        else:
            self.pkl_path = pkl_path

        self.name = pkl_path.name.split('.')[0].split('_')[0]
        self.seed = seed
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        log.debug(f'Discriminator device: {self.device}')

        warnings.filterwarnings("ignore", category=UserWarning,
                                module=".*upfirdn2d.*",
                                message=".*Failed to build CUDA kernels for upfirdn2d.*")

        with open(self.pkl_path, 'rb') as f:
            log.debug(f'pkl: {pkl_path}')
            data = legacy.load_network_pkl(f)
            self.D = data['D'].to(self.device)
            if self.device.type == 'cpu':
                # see https://github.com/NVlabs/stylegan2-ada-pytorch/issues/105
                import functools
                self.D.forward = functools.partial(self.D.forward, force_fp32=True)
            self.D.eval()

    @staticmethod
    def load_from_name(name: str) -> "StyleGANDiscriminator":
        return StyleGANDiscriminator(pkl_path=root_dir / "TrainedStyleGAN" / f"{name}.pkl")

    def calc_score(self, img: Tensor, c: Tensor | np.ndarray) -> Tensor:
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
