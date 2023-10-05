import os

import torch
from PIL import Image
from matplotlib import pyplot as plt

from sample_augment.models.generator import StyleGANGenerator, GC10_CLASSES
from sample_augment.utils import log
from sample_augment.utils.path_utils import shared_dir

from torchvision.transforms import ToPILImage
from tqdm import tqdm


def generate_instances(class_idx1=0, class_idx2=1, num_instances=100,
                       save_path=shared_dir / "generated" / "multilabel"):
    save_path.mkdir(exist_ok=True)
    # save as class name dirs
    os.makedirs(f"{save_path}/{GC10_CLASSES[class_idx1]}", exist_ok=True)
    os.makedirs(f"{save_path}/{GC10_CLASSES[class_idx2]}", exist_ok=True)
    os.makedirs(f"{save_path}/{GC10_CLASSES[class_idx1]}_{GC10_CLASSES[class_idx2]}", exist_ok=True)

    generator = StyleGANGenerator.load_from_name('unified-030_011200', seed=42)

    to_pil_image = ToPILImage()

    batch_size = 10
    num_batches = num_instances // batch_size

    for idx, class_name in [(class_idx1, GC10_CLASSES[class_idx1]),
                            (class_idx2, GC10_CLASSES[class_idx2]),
                            (None, f"{GC10_CLASSES[class_idx1]}_{GC10_CLASSES[class_idx2]}")]:
        for i in tqdm(range(num_batches)):
            c = torch.zeros((batch_size, 10))
            if idx is not None:
                c[:, idx] = 1.0
            else:
                c[:, class_idx1] = 1.0
                c[:, class_idx2] = 1.0

            with torch.no_grad():
                imgs = generator.generate(c=c)
                imgs = imgs.permute(0, 3, 1, 2)

            for j, img in enumerate(imgs):
                img_pil = to_pil_image(img.cpu())
                img_path = f"{save_path}/{class_name}/img_{i * batch_size + j}.png"
                log.info(f'saving to {img_path}')
                img_pil.save(img_path)


def multilabel_generator_plot(save_path=shared_dir / "generated/multilabel"):
    fig, axarr = plt.subplots(3, 4, figsize=(12, 9))

    # Dummy indices for each row
    indices = [0, 10, 20, 30]

    class1 = "punching_hole"
    class2 = "welding_line"
    class_both = f"{class1}_{class2}"

    # Load and plot images for class1
    for idx, col in enumerate(indices):
        img_path = os.path.join(save_path, class1, f"img_{col}.png")
        img = Image.open(img_path)
        axarr[0, idx].imshow(img)
        axarr[0, idx].axis('off')
        axarr[0, idx].set_title(f"{class1} idx: {col}")

    # Load and plot images for class2
    for idx, col in enumerate(indices):
        img_path = os.path.join(save_path, class2, f"img_{col}.png")
        img = Image.open(img_path)
        axarr[1, idx].imshow(img)
        axarr[1, idx].axis('off')
        axarr[1, idx].set_title(f"{class2} idx: {col}")

    # Load and plot images for class_both
    for idx, col in enumerate(indices):
        img_path = os.path.join(save_path, class_both, f"img_{col}.png")
        img = Image.open(img_path)
        axarr[2, idx].imshow(img)
        axarr[2, idx].axis('off')
        axarr[2, idx].set_title(f"{class_both} idx: {col}")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # multilabel_generator_plot()
    generate_instances(class_idx1=2, class_idx2=4)
