"""
Exploratory analysis of the secondary labels provided with the xml files in `lable/`.

Goals:
- have all label information in one json file
- see how many images have multiple labels
"""

import os
import json
from typing import Dict, List, Tuple
import xmltodict

from data.download_gc10 import load_gc10_if_missing
from utils.paths import resolve_project_path


def xml_to_labels(xml: Dict) -> Tuple[int, List[int]]:
    """
        returns: (primary label, [secondary labels])
    """
    # TODO
    primary_label = -1
    try:
        primary_label = int(xml['annotation']['folder'])
    except ValueError:
        assert xml['annotation']['folder'] == 'all'
        
    return primary_label, []

def main():
    load_gc10_if_missing()

    lable_directory = resolve_project_path('data/gc-10/lable')
    id_to_labels = {}

    for filename in os.listdir(lable_directory):
        # the beginning 'img' is redundant, so cut it
        image_id = filename[3:]
        with open(os.path.join(lable_directory, filename)) as xml_file:
            raw_xml = xml_file.read()
            xml_dict = xmltodict.parse(raw_xml)
        labels = xml_to_labels(xml_dict)
        id_to_labels[image_id] = labels

    interim_dir = resolve_project_path('data/interim', create=True)
    with open(f"{interim_dir}/labels.json", "w") as json_file:
        json.dump(id_to_labels, json_file)

if __name__ == '__main__':
    main()