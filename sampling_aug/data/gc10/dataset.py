import json
import os

from PIL import Image
from torch import FloatTensor, IntTensor
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, Grayscale, Compose
from tqdm import tqdm

from sampling_aug.data.gc10.download import load_gc10_if_missing
from sampling_aug.utils.paths import project_path


class GC10InMemoryDataset(Dataset):
    # TODO compare this with normal image folder dataset
    # TODO put in its own file
    # TODO load data directly on GPU
    # TODO try TensorDataset, maybe the preprocessed version of this could be saved as a single file to save on
    # -> see https://discuss.pytorch.org/t/how-to-load-all-data-into-gpu-for-training/27609/23
    def __init__(self):
        # pre-processing transforms
        # we need to take care not to have data leakage here
        # so only transforms independent of (and uninformed by) the input data are allowed.
        self.transforms = Compose([
            Resize((256, 256)),
            ToTensor(),
            Grayscale(num_output_channels=3),
            # TODO check if these normalization factors are correct for GC-10
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        load_gc10_if_missing()
        self.root_dir = project_path('data/gc-10')
        self.data = FloatTensor(1)
        self.labels = IntTensor(1)
        self.label_dict = {}

        self._read()

    def _read(self):
        # read labels.json that contains the final label info
        with open(project_path('data/interim/labels.json')) as labels_file:
            self.label_dict = json.load(labels_file)

        n = len(self.label_dict)

        self.data = FloatTensor(n, 3, 256, 256) # I would like to try training at 512, 512 as well
        self.labels = IntTensor(n)

        print("loading GC10 into RAM and doing preprocessing..", flush=True)

        img_idx = 0
        for subdir_idx in tqdm(range(1, 11), unit='class'):
            subdir = os.path.join(self.root_dir, str(subdir_idx))
            for img_filename in os.listdir(subdir):
                # get img_id to get label from label_dict
                img_id = img_filename.split('.')[0][4:]
                # save img_idx in label_dict in case we want to go from data to the actual image name
                self.label_dict[img_id]['idx'] = img_idx
                label = self.label_dict[img_id]['y']

                # apply preprocessing transforms
                # GC10 is grayscale, but I presume we need RGB data
                img = Image.open(os.path.join(subdir, img_filename)).convert('RGB')
                img_t = self.transforms(img)
                self.data[img_idx] = img_t
                self.labels[img_idx] = label
                img_idx += 1
                # if img_idx % 100 == 0:
                #     print(f"read {img_idx} images")

        print(f"done, {self.data[0, 1, :]}")

    def __len__(self):
        return self.data.shape[0]
