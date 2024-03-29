"""
GC10 exploratory data analysis (EDA)

Things I'd like to look at:
- avg intensity per class
- histogram per class
- percentage of multi-classes and which classes are likely to be secondary
- class instance counts and compare with Excel file
- look at the whole training set with a dimension reduction method such as UMAP
"""
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from sample_augment.core import step
from sample_augment.data.gc10.read_labels import GC10Labels


@step
def gc10_data_exploration(labels_artifact: GC10Labels, shared_directory: Path):
    labels = labels_artifact.labels
    # how many instances with secondary labels?
    total_count = len(labels)
    secondary_count = sum(labels[img_id]['secondary'] != [] for img_id in labels)
    print(f"ratio of images with secondary labels: {secondary_count / total_count:.2f}")

    secondary_classes = [0 for _ in range(10)]
    label_matrix = np.zeros((10, 10))

    for label in labels.values():
        primary = label['y']
        for secondary in label['secondary']:
            secondary_classes[secondary - 1] += 1
            label_matrix[primary - 1][secondary - 1] += 1

    print(label_matrix)

    classes = [
        "punching_hole",
        "welding_line",
        "crescent_gap",
        "water_spot",
        "oil_spot",
        "silk_spot",
        "inclusion",
        "rolled_pit",
        "crease",
        "waist_folding"
    ]

    fig, ax = plt.subplots()
    ax.imshow(label_matrix.T)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(classes)), labels=classes)
    ax.set_yticks(np.arange(len(classes)), labels=classes)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(label_matrix.T[i, j]),
                    ha="center", va="center", color="w")

    plt.ylabel('Primary class')
    plt.xlabel('Secondary class')
    ax.set_title("Occurrences of secondary class labels per primary class")
    fig.tight_layout()

    plt.savefig(shared_directory / 'secondary_labels.pdf', bbox_inches='tight')
    plt.show()
