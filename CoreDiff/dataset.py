import os
import os.path as osp
from glob import glob
from torch.utils.data import Dataset
import numpy as np
import torch
from functools import partial

class CTDataset(Dataset):
    def __init__(self, dataset, mode, test_id=9, dose=5, context=True):
        self.mode = mode
        self.context = context
        print(f"Using dataset: {dataset}")

        if dataset == 'custom':
            # same as above, but uses the corediff_inputs folder
            data_root = './corediff_inputs'
            patient_dirs = sorted(glob(osp.join(data_root, 'CT*')))
            patient_lists = []
            for ct_dir in patient_dirs:
                files = sorted(glob(osp.join(ct_dir, '*_triplet.npy')))
                patient_lists.extend(files)
            self.input = patient_lists

        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        print(f"Loaded {len(self.input)} samples.")

    def __getitem__(self, index):
        triplet = np.load(self.input[index]).astype(np.float32)  # shape (3, 512, 512)
        triplet = self.normalize_(triplet)
        dummy_target = np.zeros((1, 512, 512), dtype=np.float32)
        return triplet, dummy_target  # shape: (3, H, W)

    def __len__(self):
        return len(self.input)

    def normalize_(self, img, MIN_B=-1024, MAX_B=3072):
        img = img - 1024
        img[img < MIN_B] = MIN_B
        img[img > MAX_B] = MAX_B
        img = (img - MIN_B) / (MAX_B - MIN_B)
        return img

# Update your dataset_dict accordingly
dataset_dict = {
    'custom': partial(CTDataset, dataset='custom', mode='test', test_id=None, dose=None, context=True),
}
