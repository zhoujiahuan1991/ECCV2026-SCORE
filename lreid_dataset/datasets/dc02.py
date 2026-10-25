from __future__ import division, print_function, absolute_import
import os
import copy
from lreid.data_loader.incremental_datasets import IncrementalPersonReIDSamples
import re
import glob
import os.path as osp
import warnings
import numpy as np

class IncrementalSamples4dc02(IncrementalPersonReIDSamples):
    '''
    DC02 Dataset
    '''
    _junk_pids = [0, -1]
    dataset_dir = 'DeepChange_Select'
    def __init__(self, datasets_root, relabel=True, combineall=False):
        self.relabel = relabel
        self.combineall = combineall
        root = osp.join(datasets_root, self.dataset_dir)
        self.train_dir = osp.join(root, 'train_2')
        self.train_list = osp.join(root, 'train_2.txt')
        self.query_dir = osp.join(root, 'test-set-query')
        self.query_list = osp.join(root, 'test-set-query-bbox.txt')
        self.gallery_dir = osp.join(root, 'test-set-gallery')
        self.gallery_list = osp.join(root, 'test-set-gallery-bbox.txt')

        train_names = self._get_names(self.train_list)
        query_names = self._get_names(self.query_list)
        gallery_names = self._get_names(self.gallery_list)

        pid2label, clothes2label, pid2clothes = self.get_pid2label_and_clothes2label(train_names)
        train, num_train_pids, num_train_clothes = self._process_dir(self.train_dir, train_names, clothes2label, pid2label=pid2label)

        pid2label, clothes2label = self.get_pid2label_and_clothes2label(query_names, gallery_names)
        query, num_query_pids, num_query_clothes = self._process_dir(self.query_dir, query_names, clothes2label)
        gallery, num_gallery_pids, num_gallery_clothes = self._process_dir(self.gallery_dir, gallery_names, clothes2label)

        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(train, query, gallery)
    
    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def get_pid2label_and_clothes2label(self, img_names1, img_names2=None):
        if img_names2 is not None:
            img_names = img_names1 + img_names2
        else:
            img_names = img_names1

        pid_container = set()
        clothes_container = set()
        for img_name in img_names:
            names = img_name.split('.')[0].split('_')
            clothes = names[0] + names[2]
            pid = int(names[0][1:])
            pid_container.add(pid)
            clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes:label for label, clothes in enumerate(clothes_container)}

        if img_names2 is not None:
            return pid2label, clothes2label

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)
        pid2clothes = np.zeros((num_pids, num_clothes))
        for img_name in img_names:
            names = img_name.split('.')[0].split('_')
            clothes = names[0] + names[2]
            pid = int(names[0][1:])
            pid = pid2label[pid]
            clothes_id = clothes2label[clothes]
            pid2clothes[pid, clothes_id] = 1

        return pid2label, clothes2label, pid2clothes

    def _process_dir(self, home_dir, img_names, clothes2label, pid2label=None):
        dataset = []
        pid_container = set()
        clothes_container = set()
        for img_name in img_names:
            img_path = osp.join(home_dir, img_name.split(',')[0])
            names = img_name.split('.')[0].split('_')
            tracklet_id = int(img_name.split(',')[1])
            clothes = names[0] + names[2]
            clothes_id = clothes2label[clothes]
            clothes_container.add(clothes_id)
            pid = int(names[0][1:])
            pid_container.add(pid)
            camid = int(names[1][1:])
            if pid2label is not None:
                pid = pid2label[pid]
            # on DeepChange, we allow the true matches coming from the same camera 
            # but different tracklets as query following the original paper.
            # So we use tracklet_id to replace camid for each sample.
            dataset.append((img_path, pid, tracklet_id, 0))
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        return dataset, num_pids, 0
    
    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid, 0))

        return data