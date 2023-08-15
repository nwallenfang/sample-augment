import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor, Resize

from sample_augment.models.discriminator import StyleGANDiscriminator
from sample_augment.models.generator import StyleGANGenerator
from sample_augment.utils.path_utils import shared_dir, root_dir

"""
    feed some real/generated/projected data into the Discriminator and check the output probabilities
"""


def main():
    # discriminator = StyleGANDiscriminator.load_from_name('wdataaug-025_006200')
    discriminator = StyleGANDiscriminator.load_from_name('wdataaug-028_012200')
    # img = Image.open(shared_dir / 'gc10' / '01' / 'img_02_425503100_00017.jpg').convert('RGB')
    img = Image.open(shared_dir / "delete_this.jpg").convert('RGB')

    img = Resize((256, 256))(ToTensor()(img))

    c = torch.zeros((1, 10))
    c[0, 2] = 1.0
    rating = discriminator.calc_score(img, c)
    print(rating)

    # TODO ok, getting some weird results with the D scores being centered around 24
    # could feed multiple real/fake ones and check the mean
    # or check in the official implementation if we're doing everything the right way

    tensors = torch.load(root_dir / 'stylegan_train_data.pt')

    # StyleGAN expects uint8, so we'll need to do some conversions
    data = tensors[0].numpy().astype(np.uint8)
    labels = tensors[1].numpy().astype(np.float32)
    for i in range(10):
        img = data[i]
        label = labels[i]
        display_img = img.transpose(1, 2, 0)

        plt.imshow(display_img)
        plt.title(f"Inspected Image {str(label)}")
        plt.axis('off')
        plt.show()
        rand_score = discriminator.calc_score(data[i], labels[i])
        print(rand_score)


if __name__ == '__main__':
    main()
