import numpy as np
from imblearn.over_sampling import SMOTE
import os
from glob import glob
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from sample_augment.models.generator import GC10_CLASSES
from sample_augment.utils.path_utils import shared_dir


def oversample_latents(projected_dir):
    X = []
    y = []
    class_names = []

    # Go through each subdir and collect the latent vectors
    for class_name in os.listdir(projected_dir):
        class_dir = os.path.join(projected_dir, class_name)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                if file.endswith(".npz"):
                    latent_vector = np.load(os.path.join(class_dir, file))['w']
                    X.append(latent_vector.reshape(-1))
                    y.append(class_name)
            class_names.append(class_name)

    # Convert X and y to arrays
    X = np.array(X)
    y = np.array(y)

    # Apply scikit-learn Label-Encoding to class names
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y_encoded)

    # Save the augmented latent vectors back to disk
    for i in range(len(X_res)):
        class_name = le.inverse_transform([y_res[i]])[0]
        class_dir = os.path.join(projected_dir, class_name)
        file_name = os.path.join(class_dir, f"augmented_{i}.npz")
        # Reshape the vector back to original shape and save
        np.savez(file_name, w=X_res[i].reshape(1, -1))


def plot_tsne_of_latents(directory, random_state=42):
    # Get a list of all npz files in the directory
    npz_files = glob(os.path.join(directory, "*.npz"))

    # Initialize an empty list to store the latent vectors and labels
    latent_vectors = []
    labels = []

    # Loop over the npz files and load each one
    for file in npz_files:
        latent = np.load(file)['w']
        latent_vectors.append(latent)

        # Extract the class label from the filename
        filename = os.path.basename(file)
        label = next((class_name for class_name in GC10_CLASSES if class_name in filename), None)
        labels.append(label)

    latent_vectors = np.concatenate(latent_vectors)

    # Flatten the latent vectors
    latent_vectors_flattened = latent_vectors.reshape(len(latent_vectors), -1)

    # Encode the labels as integers for the scatter plot
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    tsne = TSNE(n_components=2, random_state=random_state)
    latent_vectors_2d = tsne.fit_transform(latent_vectors_flattened)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_vectors_2d[:, 0], latent_vectors_2d[:, 1], c=labels_encoded, cmap='tab10')

    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")

    for t, l in zip(legend1.texts, le.classes_):
        t.set_text(l)

    plt.title("t-SNE projection of the latent vectors")
    plt.show()


if __name__ == '__main__':
    plot_tsne_of_latents(directory=shared_dir / "projected")
