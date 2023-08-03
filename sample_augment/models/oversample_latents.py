import numpy as np
import os
from glob import glob

import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D

from sample_augment.models.generator import GC10_CLASSES, StyleGANGenerator
from sample_augment.utils import log
from sample_augment.utils.path_utils import shared_dir, root_dir
import re

from PIL import Image


def oversample_latents(projected_dir):
    X = []
    y = []
    indices = []
    class_names = []
    pattern = re.compile(r"^proj_w_([a-zA-Z_]+)_proj_\1_(\d+)\.npz$")

    # Go through each file in the dir and collect the latent vectors
    for file_name in os.listdir(projected_dir):
        if file_name.endswith(".npz"):
            # Extract class name and index from file name
            m = pattern.match(file_name)
            if m:
                class_name, index = m.groups()
                latent_vector = np.load(os.path.join(projected_dir, file_name))['w']
                X.append(latent_vector.reshape(-1))
                y.append(class_name)
                indices.append(int(index))  # keep track of index for future reference if needed
                if class_name not in class_names:
                    class_names.append(class_name)

    # Convert X, y, and index to arrays
    X = np.array(X)
    y = np.array(y)
    indices = np.array(indices)

    # Apply scikit-learn Label-Encoding to class names
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y_encoded)

    # Compute synthetic samples per parent
    synthetic_samples_per_parent = len(X_res) // len(X)

    # Repeat indices accordingly
    indices_res = np.repeat(indices, synthetic_samples_per_parent)

    # Save to file
    file_name = os.path.join(projected_dir, "augmented_latents.npz")
    np.savez(file_name, X_res=X_res, y_res=y_res, indices_res=indices_res)


def generate_classwise_w_latents(n_per_class=20):
    generator = StyleGANGenerator.load_from_name('wdataaug-025_006200')
    num_classes = 10
    w_tensor = torch.empty((n_per_class * num_classes, generator.G.num_ws, generator.G.w_dim))
    class_names = []
    for cls in range(num_classes):
        log.info(f"Generating for cls {cls}..")
        cls_label = torch.zeros((1, num_classes))
        cls_label[0][cls] = 1.0
        for i in range(n_per_class):
            w = generator.c_to_w(cls_label)
            w_tensor[cls * num_classes + i] = w
            class_names.append(GC10_CLASSES[cls])

    return w_tensor, class_names


def plot_tsne_of_latents(directory, random_state=42):
    from sklearn.manifold import TSNE

    # Get a list of all npz files in the directory
    npz_files = glob(os.path.join(directory, "*.npz"))

    # Initialize an empty list to store the latent vectors and labels
    projected_latents = []
    labels = []

    # Loop over the npz files and load each one
    for file in npz_files:
        latent = np.load(file)['w']
        projected_latents.append(latent)

        # Extract the class label from the filename
        filename = os.path.basename(file)
        label = next((class_name for class_name in GC10_CLASSES if class_name in filename), None)
        labels.append(label)

    projected_latents = np.concatenate(projected_latents)
    random_latents, random_labels = generate_classwise_w_latents()
    # Flatten the latent vectors
    latent_vectors_flattened = projected_latents.reshape(len(projected_latents), -1)
    random_latents_flattened = random_latents.reshape(len(random_latents), -1)

    # Combine labels and encode them as integers for the scatter plot
    all_labels = np.concatenate([labels, random_labels])
    all_labels_encoded = [GC10_CLASSES.index(name) for name in all_labels]
    # le = LabelEncoder()
    # all_labels_encoded = le.fit_transform(all_labels)

    # Split the encoded labels back up
    labels_encoded = all_labels_encoded[:len(labels)]
    random_labels_encoded = all_labels_encoded[len(labels):]

    tsne = TSNE(n_components=2, random_state=random_state)
    all_latents_flattened = np.concatenate([latent_vectors_flattened, random_latents_flattened])

    # Apply t-SNE on the combined data
    all_latents_2d = tsne.fit_transform(all_latents_flattened)

    # Create separate 2D arrays for the projected and random latents
    latent_vectors_2d = all_latents_2d[:len(latent_vectors_flattened)]
    random_latents_2d = all_latents_2d[len(latent_vectors_flattened):]

    plt.figure(figsize=(10, 8))
    _scatter = plt.scatter(latent_vectors_2d[:, 0], latent_vectors_2d[:, 1], c=labels_encoded, cmap='tab10')
    _scatter2 = plt.scatter(random_latents_2d[:, 0], random_latents_2d[:, 1], c=random_labels_encoded, cmap='tab10',
                           marker='x')

    # TODO need to fix projection of small classes and verify that the classes match for random and projected
    cmap = plt.get_cmap('tab10')
    # Create a custom legend
    class_legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10)
                             for i in range(len(GC10_CLASSES))]

    plt.legend(handles=class_legend_elements, labels=GC10_CLASSES, title="Classes")
    plt.title("t-SNE projection of projected latent vectors and random class-wise ones")
    plt.show()


def generate_images_from_latents(augmented_latents_file):
    # could be a StyleGANGenerator method in theory
    # maybe we could also have a single method that combines everything, from projection to smote to generating images
    generator = StyleGANGenerator(pkl_path=root_dir / 'TrainedStyleGAN' / 'wdataaug-025_006200.pkl')
    log.info(f'Generating images from augmented latents "{augmented_latents_file}"')
    data = np.load(augmented_latents_file)
    ws = data['X_res']
    ws = torch.tensor(ws, device=generator.device)  # pylint: disable=not-callable

    # Here I'm assuming the reshaped w's shape should be (1, G.num_ws, G.w_dim)
    ws = ws.view(-1, generator.G.num_ws, generator.G.w_dim)

    # TODO for the future / training, we'll need the true label vectors for these instances

    for idx, w in enumerate(ws):
        img = generator.G.synthesis(w.unsqueeze(0), noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        Image.fromarray(img[0].cpu().numpy(), 'RGB').save(shared_dir / "generated" / "smote" /
                                                          f"augmented{idx:02d}.png")


if __name__ == '__main__':
    # generate_images_from_latents(augmented_latents_file=shared_dir / 'projected/first_proj/augmented_latents.npz')
    # oversample_latents(shared_dir / "projected" / "first_proj")
    plot_tsne_of_latents(shared_dir / 'projected/')
