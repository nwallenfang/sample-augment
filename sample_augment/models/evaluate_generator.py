import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms import Resize, transforms
from torchvision.utils import make_grid

from sample_augment.models.generator import GC10_CLASSES
from sample_augment.models.train_classifier import TrainedClassifier, normalize
from sample_augment.utils import log
from sample_augment.utils.path_utils import shared_dir, root_dir
from sample_augment.utils.plot import prepare_latex_plot


def load_image(path):
    image = read_image(path)
    image = Resize((256, 256))(image)
    return image  # Returning a tensor


def load_image_for_vit_classification(path):
    # use transforms compose
    image = Image.open(path).convert("RGB")  # Opens the image file and converts to RGB
    image = transforms.ToTensor()(image)  # Converts PIL Image to PyTorch tensor
    if image.size()[0] == 1:
        image = transforms.Lambda(lambda x: x.repeat(3, 1, 1))(image)  # to RGB
    image = Resize((224, 224))(image)
    image = transforms.ConvertImageDtype(dtype=torch.float32)(image)
    image = normalize(image)
    image = image.unsqueeze(0)  # Adds an extra dimension for the batch size
    return image


def make_image_grid(real_dir: Path, fake_dir: Path, class_name: str, class_index: int, grid_size=(3, 3)):
    real_images = []
    fake_images = []

    real_class_dir = real_dir / f"{class_index:02d}"
    root, _dirs, files = next(os.walk(real_class_dir))
    for file in files:
        real_image_path = os.path.join(root, file)
        real_images.append(load_image(real_image_path))

    for root, _dirs, files in os.walk(fake_dir):
        for file in files:
            fake_image_path = os.path.join(root, file)
            filename = os.path.basename(fake_image_path)
            fake_class_name = filename.split('_seed')[0]

            if fake_class_name == class_name:
                fake_images.append(load_image(fake_image_path))

    real_indices = np.random.choice(len(real_images), size=np.prod(grid_size), replace=False)
    fake_indices = np.random.choice(len(fake_images), size=np.prod(grid_size), replace=False)

    real_images_selected = torch.stack([real_images[i] for i in real_indices])
    fake_images_selected = torch.stack([fake_images[i] for i in fake_indices])

    real_grid = make_grid(real_images_selected, nrow=grid_size[0])
    fake_grid = make_grid(fake_images_selected, nrow=grid_size[0])

    return real_grid, fake_grid


def plot_grids(real_grid, fake_grid, class_name):
    prepare_latex_plot()
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].imshow(np.transpose(real_grid.numpy(), (1, 2, 0)))
    axs[0].set_title(r'Real $\texttt{' + f'{class_name}' + r'}$ Images')
    axs[0].axis('off')

    axs[1].imshow(np.transpose(fake_grid.numpy(), (1, 2, 0)))
    axs[1].set_title(r'Fake $\texttt{' + f'{class_name}' + r'}$ Images')
    axs[1].axis('off')

    plt.savefig(shared_dir / "generator_plots" / f"grid_{class_name}.pdf", bbox_inches="tight")


def real_vs_fake_grid_classwise(real_dir=shared_dir / "gc10", fake_dir=shared_dir / "generated"):
    for class_index, class_name in enumerate(GC10_CLASSES):
        log.info(f"-- {class_name} --")
        real_grid, fake_grid = make_image_grid(real_dir, fake_dir, class_name, class_index + 1)
        plot_grids(real_grid, fake_grid, class_name)


def run_classifier_on_generated_images(generated_classname: str, modelname: str, plot=False):
    if modelname == "DenseNet":
        trained_classifier: TrainedClassifier = TrainedClassifier.from_file(
            root_dir / "TrainedClassifier/aug-05_72e4d1.json"
        )
    elif modelname == "VisionTransformer":
        trained_classifier: TrainedClassifier = TrainedClassifier.from_file(
            root_dir / "TrainedClassifier/vit-data-aug_594eee.json"
        )
    else:
        raise ValueError("unknown modelname")
    generated_imgs_dir = shared_dir / "generated"

    model = trained_classifier.model
    model.eval()  # Set the model to evaluation mode

    # Device configuration - defaults to CPU unless a GPU is available on the system
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    class_probs = []
    class_name_to_index = {name: index for index, name in enumerate(GC10_CLASSES)}

    img_batch = []
    for root, _dirs, files in os.walk(generated_imgs_dir):
        for file in files:
            class_name = file.split("_seed")[0]
            if class_name == generated_classname:
                img_path = os.path.join(root, file)
                # image transformer expects 224x224
                img = load_image_for_vit_classification(img_path).to(device)
                img_batch.append(img)

    # Convert list of tensors to a single tensor
    img_batch = torch.cat(img_batch, 0)

    with torch.no_grad():
        output = torch.nn.functional.softmax(model(img_batch),
                                             dim=1)  # Assuming the model doesn't already apply softmax

        # Get the probability of the fake class for each image in the batch.
        # Update this depending on your classifier's output format.
        fake_class_probs = output[:, class_name_to_index[generated_classname]].cpu().numpy()
        class_probs.extend(fake_class_probs)

        # Determine the most likely class for each image
        most_likely_classes = output.argmax(dim=1).cpu().numpy()
        chosen_class = class_name_to_index[generated_classname]
        colors = ['red' if class_id == chosen_class else 'blue' for class_id in most_likely_classes]

    # Now we have the probabilities, we can create a scatter plot
    if plot:
        prepare_latex_plot()
        plt.figure(figsize=(10, 6))
        plt.scatter(x=[0 for _ in range(len(class_probs))], y=class_probs, c=colors)
        plt.title('Probability')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.savefig(shared_dir / "generator_plots" / f"classifier_scores_{generated_classname}.pdf")

    return fake_class_probs


# sanity check, run on real data
def run_classifier_on_real_images(real_classname: str, modelname: str, max_count=9, plot=False):
    # TODO be lenient with labels -> see f1 score calculation  !!!
    """
    Run a classifier that was trained on the real data on the real data and look at the predictions
    """
    if modelname == "DenseNet":
        trained_classifier: TrainedClassifier = TrainedClassifier.from_file(
            root_dir / "TrainedClassifier/aug-05_72e4d1.json"
        )
    elif modelname == "VisionTransformer":
        trained_classifier: TrainedClassifier = TrainedClassifier.from_file(
            root_dir / "TrainedClassifier/vit-data-aug_594eee.json"
        )
    else:
        raise ValueError("unknown modelname")
    real_imgs_dir = shared_dir / "gc10"

    model = trained_classifier.model
    model.eval()  # Set the model to evaluation mode

    # Device configuration - defaults to CPU unless a GPU is available on the system
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    class_probs = []
    class_name_to_index = {name: index for index, name in enumerate(GC10_CLASSES)}

    img_batch = []
    real_class_dir = real_imgs_dir / f"{class_name_to_index[real_classname] + 1:02d}"
    for root, _dirs, files in os.walk(real_class_dir):
        for file in files:
            img_path = os.path.join(root, file)
            # image transformer expects 224x224
            img = load_image_for_vit_classification(img_path).to(device)
            if len(img_batch) >= max_count:
                break

            img_batch.append(img)

    # Convert list of tensors to a single tensor
    img_batch = torch.cat(img_batch, 0)

    with torch.no_grad():
        output = torch.nn.functional.softmax(model(img_batch),
                                             dim=1)  # Assuming the model doesn't already apply softmax

        # Get the probability of the real class for each image in the batch.
        real_class_probs = output[:, class_name_to_index[real_classname]].cpu().numpy()
        class_probs.extend(real_class_probs)
        most_likely_classes = output.argmax(dim=1).cpu().numpy()
        chosen_class = class_name_to_index[real_classname]
        colors = ['red' if class_id == chosen_class else 'blue' for class_id in most_likely_classes]

    if plot:
        # Now we have the probabilities, we can create a scatter plot
        prepare_latex_plot()
        plt.figure(figsize=(11, 5))
        plt.scatter(x=[0 for _ in range(len(class_probs))], y=class_probs, c=colors)
        plt.title('Probability')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.savefig(shared_dir / "generator_plots" / f"classifier_scores_real_{real_classname}.pdf")

    return real_class_probs


def plot_swarm(plot=False, from_csv=None):
    """
        Do a swarmplot with the class on the x axis to compare the distribution of classification outputs
        for real and fake images.
    """
    model_name = "VisionTransformer"

    if not from_csv:
        # Convert list of probabilities to dataframe
        real_class_probs = []
        fake_class_probs = []
        for class_name in GC10_CLASSES:
            print(f"-- {class_name} --")
            reals = run_classifier_on_real_images(class_name, model_name, plot=False)
            real_class_probs.append(reals)
            fakes = run_classifier_on_generated_images(class_name, model_name, plot=False)
            fake_class_probs.append(fakes)

        data = []
        for i, class_name in enumerate(GC10_CLASSES):
            for prob in real_class_probs[i]:
                data.append({'Class': class_name, 'Probability': prob, 'Type': 'Real'})
            for prob in fake_class_probs[i]:
                data.append({'Class': class_name, 'Probability': prob, 'Type': 'Fake'})
        df = pd.DataFrame(data)
    else:
        # from csv
        df = pd.read_csv(from_csv)

    if plot:
        # Create a swarmplot
        prepare_latex_plot()
        plt.figure(figsize=(10, 5))
        sns.violinplot(x='Class', y='Probability', hue='Type', data=df, split=True, cut=0, bw=0.2, inner=None,
                       scale="count", alpha=0.7)
        # sns.swarmplot(x='Class', y='Probability', hue='Type', data=df, dodge=True,
        #               size=4)
        plt.xticks(rotation=90)
        plt.ylabel(f"{model_name} Classification Probability")
        plt.title(model_name)

        plt.savefig(shared_dir / "generator_plots" / f"classwise_classification_probs_{model_name}.pdf",
                    bbox_inches="tight")

    df.to_csv(shared_dir / "generator_plots" / f"classwise_classification_probs_{model_name}.csv")


if __name__ == '__main__':
    # plot_swarm(plot=True, from_csv=shared_dir / r"generator_plots\classwise_classification_probs_VisionTransformer.csv")
    # run_classifier_on_real_images("welding_line")
    # run_classifier_on_generated_images("welding_line")
    real_vs_fake_grid_classwise()

# TODO UMAP projection in classifier activation space :)
