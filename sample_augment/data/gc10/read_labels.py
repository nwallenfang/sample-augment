"""
Exploratory analysis of the labels provided with the xml files in `lable/` compared to the label information
that is given by the subdirectory the images are in.
The final result of this script is a combined label file `labels.json`.
"""
import collections
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
import xmltodict

from sample_augment.core import step, Artifact
from sample_augment.data.gc10.download_gc10 import GC10Folder
# from sample_augment.data.gc10.download import load_gc10_if_missing
from sample_augment.utils.paths import project_path
from sample_augment.utils.plot import show_image

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

    for filename in os.listdir(label_directory):
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


def read_image_dir_labels(image_dir: Path) -> Dict:
    """
        Create interim/labels_dir.json. Get label info from the subdirectories
    """
    image_dir_labels = {}
    for label in range(1, 11):
        for filename in os.listdir(image_dir / str(label)):
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
        path = project_path(f"data/gc10/{dir_label}/img_{name}.jpg")
        img = matplotlib.image.imread(path)

        secondary_labels = [LABEL_TO_NAME[label] for label in xml_labels[name]['secondary']]
        img_text = f"dir: {LABEL_TO_NAME[dir_labels[name]]}, xml: {LABEL_TO_NAME[xml_labels[name]['y']]}\n \
         secondary: {secondary_labels}"
        show_image(img, title=name, text=img_text,
                   save_path=project_path(f'reports/figures/label-exploration/mismatch_{name}.png'))
        mismatch_index += 1


""" manually assigned labels to the 5 images that have conflicting labels in directory/xml-file """
MANUAL_MISMATCH_LABELS = {
    # inclusion is fitting, I can't see any rolled pit features as shown in the Excel file
    "02_425392000_00984": 7,
    # could be rolled_pit or welding_line. I put it as rolled pit since we have fewer of those.
    "06_425505500_00052": 8,
    # I can't see a silk spot. punching_hole fits, welding line would fit as well
    "07_425391800_00054": 1,
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


@step
def construct_processed_labels(gc10: GC10Folder) -> GC10Labels:
    """
        combine xml labels and dir labels to construct the final label file
    """

    xml_labels = read_xml_labels(gc10.label_dir)
    dir_labels = read_image_dir_labels(gc10.image_dir)

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

    return GC10Labels(labels=labels)
