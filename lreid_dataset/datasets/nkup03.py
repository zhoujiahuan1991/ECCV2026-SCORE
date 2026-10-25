from __future__ import division, print_function, absolute_import
import os
import copy
from lreid.data_loader.incremental_datasets import IncrementalPersonReIDSamples
import re
import glob
import os.path as osp
import warnings
import numpy as np

class IncrementalSamples4nkup03(IncrementalPersonReIDSamples):
    '''
    NKUP01 Dataset
    '''
    _junk_pids = [0, -1]
    dataset_dir = 'NKUP'
    def __init__(self, datasets_root, relabel=True, combineall=False):
        self.relabel = relabel
        self.combineall = combineall
        root = osp.join(datasets_root, self.dataset_dir)
        self.train_dir = osp.join(root, 'train_3')
        self.query_dir = osp.join(root, 'bounding_box_test', 'query')
        self.gallery_dir_same = osp.join(root, 'bounding_box_test', 'gallery_sameapp')
        self.gallery_dir_cros1 = osp.join(root, 'bounding_box_test', 'gallery_crossapp_moderate')
        self.gallery_dir_cros2 = osp.join(root, 'bounding_box_test', 'gallery_crossapp_dramatic')

        train_names = glob.glob(self.train_dir+"/*/*/*")
        query_names = glob.glob(self.query_dir+"/*/*/*")
        gallery_names_same = glob.glob(self.gallery_dir_same+"/*/*/*")
        gallery_names_cros1 = glob.glob(self.gallery_dir_cros1+"/*/*/*")
        gallery_names_cros2 = glob.glob(self.gallery_dir_cros2+"/*/*/*")

        train = self._process_dir(self.train_dir, train_names, relabel=True)
        query = self._process_dir(self.query_dir, query_names, relabel=False)

        gallery_same = self._process_dir(self.gallery_dir_same, gallery_names_same, relabel=False)
        gallery_cros1 = self._process_dir(self.gallery_dir_cros1, gallery_names_cros1, relabel=False)
        gallery_cros2 = self._process_dir(self.gallery_dir_cros2, gallery_names_cros2, relabel=False)

        gallery = gallery_cros1 + gallery_cros2
        
        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(train, query, gallery)
    
    def _process_dir(self, dir_path, list_path, relabel=False):
        dataset = []
        pid_container_orig = set()
        pid_container_after = set()

        for img_idx, img_info in enumerate(list_path):
            pid = int(img_info.split('/')[-1].split('_')[0])
            pid_container_orig.add(pid)
        pid_container_orig = sorted(pid_container_orig)
        pid2label = {pid: label for label, pid in enumerate(pid_container_orig)}

        dataset = []
        for img_info in list_path:
            pid = int(img_info.split('/')[-1].split('_')[0])
            if relabel: pid = pid2label[pid]
            camid = int(img_info.split('/')[-1].split('_')[2].split('C')[1].split('S')[0])
            # print (img_info, ' ', pid, ' ', camid)
            pid_container_after.add(camid)
            dataset.append((img_info, pid, camid, 0))
        
        return dataset
