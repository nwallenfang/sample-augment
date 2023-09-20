import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sample_augment.models.evaluate_baseline import figures_dir
from sample_augment.utils.path_utils import shared_dir


def load_images_from_class(directory, class_name, num_images=8):
    class_path = os.path.join(directory, class_name)
    images = []
    if os.path.isdir(class_path):
        class_image_names = [img_name for img_name in sorted(os.listdir(class_path))
                             if img_name.endswith(('.png', '.jpg'))][:num_images]
        images = [Image.open(os.path.join(class_path, img_name)) for img_name in class_image_names]
    return images


selected_class = "oil_spot"
random_images = load_images_from_class(shared_dir / "generated" / "wdataaug-028_012200", selected_class)
handpicked_images = load_images_from_class(shared_dir / "generated" / "handpicked-wdataaug-028_012200", selected_class)
handpicked_indices = [3, 4]  # specify which indices from random_images are handpicked


# Plot images
def plot_images(images, title, handpicked_indices, num_rows=2, num_columns=4):
    for index, image in enumerate(images):
        ax = plt.subplot(num_rows, num_columns, index + 1)
        plt.imshow(image)
        plt.axis("off")

        # Highlight handpicked images
        if index in handpicked_indices:
            rect = Rectangle((0,0),1,1, transform=ax.transAxes, edgecolor='Goldenrod', facecolor='none', linewidth=5)
            ax.add_patch(rect)


# Main plotting
plt.figure(figsize=(6, 3))  # Adjust the figure size
plot_images(random_images, f"Random Sampling of {selected_class} with Handpicked Highlighted", handpicked_indices)

plt.tight_layout()
plt.savefig(figures_dir / "random_vs_handpicked.pdf", bbox_inches="tight")
