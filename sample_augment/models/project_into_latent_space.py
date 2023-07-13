import sys
from pathlib import Path
from typing import Union

from sample_augment.models.stylegan2.projector import run_projection
from sample_augment.utils import log
from sample_augment.utils.path_utils import root_dir, shared_dir

from sample_augment.models.generator import GC10_CLASSES



def project(pkl_path: Path, image_path: Union[str, Path], target_class: int):
    if isinstance(image_path, str):
        image_path = Path(image_path)
    outdir = shared_dir / "projected"
    image_id = image_path.name.split('.')[0].split('img_')[1]
    # generator = StyleGANGenerator(pkl_path=pkl_path)
    run_projection(
        str(pkl_path),
        str(image_path),
        target_class=target_class,
        target_classname=GC10_CLASSES[target_class],
        projected_fname=image_id,
        outdir=str(outdir),
        save_video=False,
        seed=1,
        num_steps=1000
    )


if __name__ == '__main__':
    # pkl_path = root_dir / 'TrainedStyleGAN/network-snapshot-001000.pkl'
    # generator = StyleGANGenerator(pkl_path=pkl_path)
    # TODO for each class, pick a random one
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = shared_dir / "gc10" / "01" / "img_02_425506300_00018.jpg"
        log.info(f"using default img_path {img_path}")

    prefix  = shared_dir / "gc10"
    img_paths = {
        0: "01/img_03_4406742300_00001.jpg", # punching_hole
        1: "02/img_02_3402616900_00001.jpg", # welding line
        2: "03/img_01_425008500_00874.jpg",  # crescent gap
        7: "08/img_03_4402329000_00001.jpg"  # waist folding
    }
    for target_class, img_path in img_paths.items():
        log.info(f'--- {GC10_CLASSES[target_class]} ---')
        project(pkl_path=root_dir / 'TrainedStyleGAN/apa-020_004400.pkl', image_path=prefix / img_path, target_class=target_class)
