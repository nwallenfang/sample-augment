"""
Exploratory analysis of the labels provided with the xml files in `lable/` compared to the label information
that is given by the subdirectory the images are in.
The final result of this script is a combined label file `labels.json`.
"""
import collections
import os
from copy import copy
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import seaborn as sns
import xmltodict
from matplotlib import pyplot as plt, gridspec

from sample_augment.core import step, Artifact
from sample_augment.data.gc10.download_gc10 import GC10Folder
from sample_augment.models.generator import GC10_CLASSES
from sample_augment.utils import log
# from sample_augment.data.gc10.download import load_gc10_if_missing
from sample_augment.utils.path_utils import shared_dir
from sample_augment.utils.plot import show_image_tensor, prepare_latex_plot, CREATE_PLOTS

# from sample_augment.data/gc10/Defect Descriptions.xlsx, which also contains some example images
LABEL_TO_NAME = {
    1: "punching_hole",
    2: "welding_line",
    3: "crescent_gap",
    4: "water_spot",
    5: "oil_spot",
    6: "silk_spot",
    7: "inclusion",
    8: "rolled_pit",
    9: "crease",
    10: "waist_folding"
}


def xml_to_labels(xml: Dict) -> Dict:
    """
        returns: (primary label, [secondary labels])
    """
    try:
        primary_label = int(xml['annotation']['folder'])
    except ValueError:
        assert xml['annotation']['folder'] == 'all'
        primary_label = 'all'

    secondary = []
    # read secondary label(s)
    if 'object' in xml['annotation']:
        objects = xml['annotation']['object']
        if type(objects) is list:  # multiple objects
            for obj in xml['annotation']['object']:
                try:
                    label = int(obj['name'].split('_')[0])
                    secondary.append(label)
                except ValueError:
                    pass  # mislabelled instance
        elif type(objects) in (collections.OrderedDict, dict):  # one object
            label = int(objects['name'].split('_')[0])  # first character is label index
            secondary.append(label)
        else:
            raise ValueError(f'weird objects tag in xml {type(objects)}')

    return {'y': primary_label, 'secondary': secondary}


def read_xml_labels(label_directory: Path) -> Dict:
    """
        Create interim/xml_labels.json file. Get label info from the xml files
    """
    xml_labels = {}

    for filename in next(os.walk(label_directory))[2]:
        # the beginning 'img_' is redundant, so cut it
        image_id = filename.split('.')[0][4:]
        with open(os.path.join(label_directory, filename)) as xml_file:
            raw_xml = xml_file.read()
            xml_dict = xmltodict.parse(raw_xml)
        labels = xml_to_labels(xml_dict)
        xml_labels[image_id] = labels

    # some xml files contain the directory 'all' as their label
    # replace these labels with their respective directory
    # while we're at it, we should perform some sanity checks:
    # see if the xml labels and the subdirectories match,
    # see if the number of class instances match
    # there are 2294 instances ( len(id_to_labels) )
    # that matches number of xml files, but doesn't match number of image files which is 2300
    # so there are 6 images missing

    # interim_dir = project_path('data/interim', create=True)
    # with open(f"{interim_dir}/xml_labels.json", "w") as json_file:
    #     json.dump(id_to_labels, json_file, indent=2)

    return xml_labels


def read_image_dir_labels(gc10: GC10Folder) -> Dict:
    """
        Create interim/labels_dir.json. Get label info from the subdirectories
    """
    image_dir_labels = {}
    _, dirnames, _ = next(os.walk(gc10.image_dir))
    num_classes = int(sorted(dirnames)[-1])
    for label in range(1, num_classes + 1):
        for filename in next(os.walk(gc10.image_dir / f"{label:02d}"))[2]:
            image_id = filename.split('.')[0][4:]  # cut out 'img_'
            image_dir_labels[image_id] = label

    return image_dir_labels


def compare_dir_and_xml_labels(xml_labels: Dict, dir_labels: Dict):
    # print(f"xml size: {len(xml_labels)}")
    # print(f"dir size: {len(dir_labels)}")
    # since there are more dir labels I'm expecting every xml label to exist in dir labels
    for image_id in xml_labels:
        assert image_id in dir_labels

    # find the instances that ar not contained in xml
    # not_in_xml = list(filter(lambda name: name not in xml_labels, dir_labels))
    # not_in_xml_dir_labels = [dir_labels[name] for name in not_in_xml]
    # print(not_in_xml)
    # print(not_in_xml_dir_labels)

    # check if for the others, the primary ids match
    mismatches = []
    for name in dir_labels:
        if name in xml_labels:
            if not dir_labels[name] == xml_labels[name]['y']:
                # print(f"{name}: dir_id = {dir_labels[name]}, xml_id = {xml_labels[name]['y']}")
                mismatches.append(name)

    # most mismatches are from the xml_id containing all, which is no big problem
    # there are a couple of real mismatches though, and it would be interesting to look at those
    mismatch_index = 1
    for name in mismatches:
        dir_label, xml_label = dir_labels[name], xml_labels[name]['y']
        if xml_label == 'all':
            continue  # skip these for now
        path = shared_dir / f"gc10/{dir_label}/img_{name}.jpg"
        img = matplotlib.image.imread(path)

        secondary_labels = [LABEL_TO_NAME[label] for label in xml_labels[name]['secondary']]
        img_text = f"dir: {LABEL_TO_NAME[dir_labels[name]]}, xml: {LABEL_TO_NAME[xml_labels[name]['y']]}\n \
         secondary: {secondary_labels}"
        show_image_tensor(img, title=name, text=img_text,
                          save_path=shared_dir / f'figures/label-exploration/mismatch_{name}.png')
        mismatch_index += 1


""" manually assigned labels to the 5 images that have conflicting labels in directory/xml-file """
MANUAL_MISMATCH_LABELS = {
    # inclusion is fitting, I can't see any rolled pit features as shown in the Excel file
    "02_425392000_00984": 7,
    # could be rolled_pit or welding_line. I put it as rolled pit since we have fewer of those.
    "06_425505500_00052": 8,
    # I can't see a silk spot. punching_hole fits, welding line would fit as well
    "07_425391800_00054": 2,
    # both punching hole and welding line
    "07_425502900_00052": 2,
    # the image has light wait folding, but also a welding line and a punching hole
    # this one will likely be hard to classify
    "07_436163600_01161": 10,
}


def remove_duplicates(primary: int, secondary: List[int]):
    """
        remove duplicate secondary labels and remove the primary label from the secondary labels
    """
    secondary = list(set(secondary))
    if primary in secondary:
        secondary.remove(primary)
    return secondary


class GC10Labels(Artifact):
    labels: Dict
    number_of_classes: int
    class_names = [
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


class SanitizedGC10Labels(GC10Labels):
    pass


@step
def construct_processed_labels(gc10: GC10Folder) -> GC10Labels:
    """
        combine xml labels and dir labels to construct the final label file
    """

    xml_labels = read_xml_labels(gc10.label_dir)
    dir_labels = read_image_dir_labels(gc10)

    labels = dir_labels.copy()

    for img_id in dir_labels:
        if img_id not in xml_labels or xml_labels[img_id]['y'] == 'all':
            labels[img_id] = {
                'y': dir_labels[img_id], 'secondary': []
            }
        elif dir_labels[img_id] != xml_labels[img_id]['y']:
            # one of 5 mismatches, look up the label
            y = MANUAL_MISMATCH_LABELS[img_id]
            secondary = remove_duplicates(y, xml_labels[img_id]['secondary'])
            labels[img_id] = {
                'y': y, 'secondary': secondary
            }
        else:
            # default case, take primary and secondary from xml
            y, secondary = xml_labels[img_id]['y'], xml_labels[img_id]['secondary']
            secondary = remove_duplicates(y, secondary)
            labels[img_id] = {
                'y': y, 'secondary': secondary
            }

    _, dirnames, _ = next(os.walk(gc10.image_dir))
    num_classes = int(sorted(dirnames)[-1])
    return GC10Labels(labels=labels, number_of_classes=num_classes)


def class_wise_label_count_ratio(labels):
    # Create a dictionary where keys are classes and values are lists of instances
    class_instances = collections.defaultdict(list)

    for img_id, label_info in labels.items():
        primary_class = label_info['y']
        all_class_labels = [primary_class] + label_info['secondary']
        class_instances[primary_class].append(all_class_labels)

    # Calculate the ratio of instances having only one label for each class
    class_ratios = {}

    for class_label, instances in class_instances.items():
        ratio_single_label = sum(len(instance) == 1 for instance in instances) / len(instances)
        class_ratios[class_label] = ratio_single_label

    class_ratios = {int(k): v for k, v in class_ratios.items()}

    # Print the ratio for each class
    # for class_label, ratio in sorted(class_ratios.items()):
    #     print(f"Class {class_label}, single ratio: {ratio:.2f}")

    # primary and secondary
    all_labels = {img_id: labels[img_id]['secondary'] + [labels[img_id]['y']] for img_id in labels}

    # ratio of instances having only one label
    # ratio_single_label = sum([len(instance) == 1 for instance in all_labels.values()]) / len(all_labels)
    # print(f"ratio of instances with just a single label: {ratio_single_label:.2f}")


def ratio_of_secondary_classes(labels: Dict, primary_class_idx: int, class_names: List[str]):
    class_labels = {img_id: label['secondary'] for img_id, label in labels.items() if primary_class_idx == label['y']}
    secondary_counter = collections.defaultdict(int)

    for img_id, secondary in class_labels.items():
        for sec in secondary:
            secondary_counter[class_names[sec]] += 1

    # Compute proportion of instances without a secondary class
    no_secondary_count = len([img_id for img_id, sec in class_labels.items() if not sec])
    total_count = len(class_labels)
    no_secondary_ratio = no_secondary_count / total_count

    # Include the no_secondary_ratio in the secondary_ratios dictionary
    secondary_ratios = {k: v / total_count for k, v in secondary_counter.items()}
    secondary_ratios[class_names[primary_class_idx]] = no_secondary_ratio

    return secondary_ratios


def assign_instance_to(new_class_idx: int, img_id: str, all_labels: Dict):
    labels_tmp: List[int] = copy(all_labels[img_id])
    labels_tmp.remove(new_class_idx)  # rm welding_line
    return {
        'y': new_class_idx,
        'secondary': labels_tmp
    }


def calc_relative_distributions(all_labels):
    distributions = []
    absolute_counts = []
    for class_idx, _row_name in enumerate(GC10_CLASSES):
        instances_with_class = [(instance_id, labels) for instance_id, labels in all_labels.items()
                                if class_idx in labels]
        n_instances = len(instances_with_class)
        # how frequently (relatively) other classes co-occur with this one
        class_co_occurrences = []
        abs_co_occurences = []

        for co_class_idx, _col_name in enumerate(GC10_CLASSES):
            if class_idx == co_class_idx:
                # special case: ratio of instances having only that label
                count = sum([1 for (instance_id, labels) in instances_with_class if len(labels) == 1])
            else:
                # determine the ratio of col_class occurance in all instances that have row_class
                count = sum([1 for (instance_id, labels) in instances_with_class if co_class_idx in labels])

            ratio = count / n_instances

            class_co_occurrences.append(ratio)
            abs_co_occurences.append(count)

        distributions.append(class_co_occurrences)
        absolute_counts.append(abs_co_occurences)

    return distributions, absolute_counts


@step
def sanitize_labels(gc10_labels: GC10Labels) -> SanitizedGC10Labels:
    class_name_to_idx = {name: idx for idx, name in enumerate(gc10_labels.class_names)}
    # sanity check: number of class instances
    labels: Dict = copy(gc10_labels.labels)
    # make labels 0-indexed to have it be the same as pytorch Dataset
    labels = {img_id: {'y': label['y'] - 1, 'secondary': [sec - 1 for sec in label['secondary']]}
              for img_id, label in labels.items()}

    # basically ignore the distinction of y and secondary, it's all the same to us
    all_labels = {img_id: labels[img_id]['secondary'] + [labels[img_id]['y']] for img_id in labels}

    # sanitized_labels = copy(labels)

    # def move_combination_into(combined: List[str], into_class: str):
    #     for img_id, label in sanitized_labels.items():
    #         if all([(class_name_to_idx[class_name] in all_labels[img_id]) for class_name in combined]):
    #             sanitized_labels[img_id] = assign_instance_to(class_name_to_idx[into_class], img_id, all_labels)

    # when instance is labelled welding_line and punching_hole, put it into welding line!
    # move_combination_into(['punching_hole', 'welding_line'], into_class='welding_line')

    # _ratio = ratio_of_secondary_classes(sanitized_labels, class_name_to_idx['punching_hole'], gc10_labels.class_names)

    from itertools import chain
    class_counts = collections.Counter(chain.from_iterable(all_labels.values()))
    log.info(class_counts)

    if CREATE_PLOTS:
        # Heatmap data generation
        class_names = [" ".join(class_name.split("_")) for class_name in gc10_labels.class_names]
        total_counts = []
        heatmap_data, absolute_counts = calc_relative_distributions(all_labels)
        for primary_class_idx in range(len(class_names)):
            #     # Append the distribution to heatmap data, while making sure each secondary class has an entry
            #     heatmap_data.append([round(secondary_distribution.get(class_name, 0), 2) for class_name in class_names])
            # Calculate the total counts for each primary class
            secondary_distribution = ratio_of_secondary_classes(labels, primary_class_idx, class_names)
            total_counts.append(sum([val for val in secondary_distribution.values()]))

        class_counts = dict(class_counts)
        total_counts = [class_counts[i] for i in range(10)]

        print(total_counts)
        # Then, you can call the heatmap plotting function
        plot_heatmap(heatmap_data, absolute_counts, class_names, class_names, total_counts)

    return SanitizedGC10Labels(labels=labels, number_of_classes=gc10_labels.number_of_classes)


def plot_heatmap(distributions, absolute_counts, primary_classes, secondary_classes, total_counts):
    import pandas as pd

    df = pd.DataFrame(distributions, index=primary_classes, columns=secondary_classes)
    # df_abs = pd.DataFrame(absolute_counts, index=primary_classes, columns=secondary_classes)

    prepare_latex_plot()
    _fig = plt.figure(figsize=(.75 * 8, .75 * 5))
    # Define the grid space
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])

    # Define two subplots
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    def custom_annotation(val):
        if val == 0:
            return '0'
        if 0 < val < 0.01:
            return r'\tiny{$<\!0.01$}'
        else:
            return f'{val:.2f}'

    # Create a heatmap on the first subplot
    _hmap = sns.heatmap(df, annot=True, cmap='YlGnBu', ax=ax0, cbar=False)

    # for text, value in zip(_hmap.texts, df.values.flatten()):
    #     text.set_text(value)

    for text in _hmap.texts:
        text.set_text(custom_annotation(float(text.get_text())))
    ax0.set_xlabel('Co-Occurring Class')
    ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45, ha="right")
    ax0.set_ylabel('Class')
    # ax0.set_title("Relative Distribution of Secondary Labels")

    # Display total counts on the second subplot
    # seaborn heatmap and bar chart have inverted y axis
    ax1.barh(list(range(len(total_counts))), total_counts[::-1], color='lightgray')
    ax1.grid(linestyle='dotted')
    ax1.set_xlabel('No. of Instances')
    xticks = [100, 300, 500, 700]

    # Sort the x-ticks
    xticks = np.sort(xticks)

    print(np.max(total_counts))
    # Set the x-ticks
    ax1.set_xticks(xticks)
    ax1.get_yaxis().set_visible(False)  # Hide the y-axis labels
    ax1.set_ylim(-0.5, len(total_counts) - 0.5)  # Adjust y limits to match the heatmap
    ax1.set_xlim(0, np.max(total_counts))
    plt.subplots_adjust(wspace=0.10)
    # plt.tight_layout()
    plt.savefig(shared_dir / "figures/label-exploration" / "secondary_labels.pdf", bbox_inches='tight',
                pad_inches=0.03)
