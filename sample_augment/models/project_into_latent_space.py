import sys
from pathlib import Path
from typing import Union

from sample_augment.models.stylegan2.projector import run_projection
from sample_augment.utils import log
from sample_augment.utils.path_utils import root_dir, shared_dir


def project(image_path: Union[str, Path]):
    if isinstance(image_path, Path):
        image_path = str(image_path)
    pkl_path = root_dir / 'TrainedStyleGAN/network-snapshot-001000.pkl'
    outdir = shared_dir / "projected"
    # generator = StyleGANGenerator(pkl_path=pkl_path)
    run_projection(
        str(pkl_path),
        image_path,
        target_class=0,
        outdir=str(outdir),
        save_video=False,
        seed=1,
        num_steps=1000
    )


if __name__ == '__main__':
    # pkl_path = root_dir / 'TrainedStyleGAN/network-snapshot-001000.pkl'
    # generator = StyleGANGenerator(pkl_path=pkl_path)
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = shared_dir / "gc10" / "01" / "img_02_425506300_00018.jpg"
        log.info(f"using default img_path {img_path}")

    project(image_path=img_path)
