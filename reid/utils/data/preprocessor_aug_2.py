from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, transform_basic=None, transform_strong=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform_weak = transform
        self.transform_basic = transform_basic
        self.transform_strong = transform_strong

        # Only return dual views when the caller explicitly provides a strong/basic transform.
        self.dual_view = (self.transform_strong is not None) or (self.transform_basic is not None)

        # Fallback order: explicit strong -> legacy basic -> reuse weak transform
        if self.transform_strong is None and self.transform_basic is not None:
            self.transform_strong = self.transform_basic
        if self.transform_strong is None:
            self.transform_strong = self.transform_weak

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        # print(self.dataset)
        try:
            fname, pid, camid, domain = self.dataset[index]
        except:
            fname, pid, camid, domain, _ = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform_weak is not None:
            img_weak = self.transform_weak(img)
        else:
            img_weak = img

        if self.dual_view:
            if self.transform_strong is not None:
                img_strong = self.transform_strong(img)
            else:
                img_strong = img_weak
            return img_weak, img_strong, fname, pid, camid, domain

        return img_weak, fname, pid, camid, domain
