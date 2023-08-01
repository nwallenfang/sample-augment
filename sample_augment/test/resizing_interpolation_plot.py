from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor

from sample_augment.utils.path_utils import shared_dir

"""
    Test the different interpolation schemes for downsampling our high-res images.
    Use a silk spot instance as a representative for image that will be hard to scale while
    preserving its features. Have a close-up of the scaled img.
"""


def main():
    image = Image.open(shared_dir / "gc10/06/img_01_425005700_00154.jpg")
    tensor = ToTensor()(image)
    original_image_size = image.size
    new_size = (256, 256)

    # Apply different resampling filters and convert to tensor
    resized_bilinear = ToTensor()(image.resize(new_size, Image.BILINEAR))
    resized_bicubic = ToTensor()(image.resize(new_size, Image.BICUBIC))
    resized_lanczos = ToTensor()(image.resize(new_size, Image.LANCZOS))

    # Select a subsection of the images to zoom in on
    original_zoom_region = (1000, 500, 1300, 800)

    # Calculate the scale factors
    scale_factor_x = new_size[0] / original_image_size[0]
    scale_factor_y = new_size[1] / original_image_size[1]

    # Calculate the equivalent zoom region in the resized image
    zoom_region = (int(original_zoom_region[0] * scale_factor_x),
                   int(original_zoom_region[1] * scale_factor_y),
                   int(original_zoom_region[2] * scale_factor_x),
                   int(original_zoom_region[3] * scale_factor_y))

    zoomed_bilinear = resized_bilinear[:, zoom_region[1]:zoom_region[3], zoom_region[0]:zoom_region[2]]
    zoomed_bicubic = resized_bicubic[:, zoom_region[1]:zoom_region[3], zoom_region[0]:zoom_region[2]]
    zoomed_lanczos = resized_lanczos[:, zoom_region[1]:zoom_region[3], zoom_region[0]:zoom_region[2]]

    # Crop a square region from the original image and resize it to the size of the cropped region from the resized image
    original_zoom_region_square = (original_zoom_region[0], original_zoom_region[1],
                                   original_zoom_region[0] + (zoom_region[3] - zoom_region[1]),
                                   original_zoom_region[1] + (zoom_region[2] - zoom_region[0]))

    zoomed_original = ToTensor()(image.crop(original_zoom_region_square).resize(
        (zoom_region[3] - zoom_region[1], zoom_region[2] - zoom_region[0])))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(zoomed_bilinear.permute(1, 2, 0), cmap='gray')
    axes[0].set_title('Bilinear')
    axes[1].imshow(zoomed_bicubic.permute(1, 2, 0), cmap='gray')
    axes[1].set_title('Bicubic')
    axes[2].imshow(zoomed_lanczos.permute(1, 2, 0), cmap='gray')
    axes[2].set_title('Lanczos')
    axes[3].imshow(zoomed_original.permute(1, 2, 0), cmap='gray')
    axes[3].set_title('Original')

    for j in range(4):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
