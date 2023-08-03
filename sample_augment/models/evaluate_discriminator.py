from sample_augment.models.generator import StyleGANGenerator
from PIL import Image
from sample_augment.utils.path_utils import shared_dir
from torchvision.transforms import ToTensor, Resize
import torch

"""
    feed some real/generated/projected data into the Discriminator and check the output probabilities
"""


def main():
    generator = StyleGANGenerator.load_from_name('wdataaug-028_012200')
    # img = Image.open(shared_dir / 'gc10' / '01' / 'img_02_425503100_00017.jpg').convert('RGB')
    img = Image.open(r"H:\thesis\sample-augment\data\shared\generated\smote\augmented73.png").convert('RGB')

    img = Resize((256, 256))(ToTensor()(img))

    c = torch.zeros((1, 10))
    c[0, 4] = 1.0
    rating = generator.img_into_discriminator(img, c)
    print(rating)

    # TODO ok, getting some weird results with the D scores being centered around 24
    # could feed multiple real/fake ones and check the mean
    # or check in the official implementation if we're doing everything the right way


if __name__ == '__main__':
    main()
