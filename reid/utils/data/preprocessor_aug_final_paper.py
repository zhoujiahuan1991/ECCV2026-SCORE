from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, transform_strong=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.transform_strong = transform_strong

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        # print(self.dataset)
        flag = 0

        try:
            fname, pid, camid, domain = self.dataset[index]
        except:
            fname, pid, camid, domain, domain2 = self.dataset[index]
            flag = 1
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img_origin = Image.open(fpath).convert('RGB')

        img = self.transform(img_origin) if self.transform is not None else img_origin

        if self.transform_strong is not None:
            img_strong = self.transform_strong(img_origin)
            return img, img_strong, fname, pid, camid, domain

        # if flag == 1:
        #     return img, fname, pid, camid, domain, domain2
        return img, fname, pid, camid, domain
