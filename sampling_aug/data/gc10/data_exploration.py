"""
GC10 exploratory data analysis (EDA)

Things I'd like to look at:
- avg intensity per class
- histogram per class
- percentage of multi-classes and which classes are likely to be secondary
- class instance counts and compare with Excel file
- look at the whole training set with a dimension reduction method such as UMAP
"""

import json

import numpy as np
from matplotlib import pyplot as plt

from sampling_aug.utils.paths import resolve_project_path


def main():
    labels = {}
    with open(resolve_project_path('data/interim/labels.json')) as label_file:
        labels = json.load(label_file)

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
    im = ax.imshow(label_matrix.T)

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

    plt.savefig(resolve_project_path('reports/figures/label-exploration/secondary_labels.pdf'))
    plt.show()


if __name__ == '__main__':
    main()
