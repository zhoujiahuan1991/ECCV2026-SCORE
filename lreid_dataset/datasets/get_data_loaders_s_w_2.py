# import torchvision.transforms as T
import copy
import os.path
import os
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from reid.utils.feature_tools import *
import lreid_dataset.datasets as datasets
from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.data import IterLoader

from reid.utils.data.preprocessor_aug_2 import Preprocessor
from PIL import ImageOps, ImageFilter
import PIL.ImageDraw
from .ShrinkMatchTransfer import RandAugmentMC


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_data(name, data_dir, height, width, batch_size, workers, num_instances, select_num=0):
    # root = osp.join(data_dir, name)
    root = data_dir
    
    dataset = datasets.create(name, root)
    '''select some persons for training'''
    if select_num > 0:
        train = []
        for instance in dataset.train:
            if instance[1] < select_num:
                # new_id=id_2_id[instance[1]]
                train.append((instance[0], instance[1], instance[2], instance[3]))

        dataset.train = train
        unique_pids = {pid for _, pid, _, _ in train}
        dataset.num_train_pids = len(unique_pids)
        dataset.num_train_imgs = len(train)

   

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = sorted(dataset.train)

    iters = int(len(train_set) / batch_size)
    num_classes = dataset.num_train_pids

    train_transformer_weak = transforms.Compose([
        transforms.Resize((height, width), interpolation=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop((height, width)),
        transforms.ToTensor(),
        normalizer,
    transforms.RandomErasing(p=0.5, value=[0.485, 0.456, 0.406])
    ])

    train_transformer_strong = transforms.Compose([
        transforms.Resize((height, width), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=(height, width), padding=int(width * 0.125), padding_mode='reflect'),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        normalizer,
    transforms.RandomErasing(p=0.5, value=[0.485, 0.456, 0.406])
    ])


    test_transformer = transforms.Compose([
        transforms.Resize((height, width), interpolation=3),
        transforms.ToTensor(),
        normalizer
    ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None


    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer_weak, transform_strong=train_transformer_strong),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)

    init_loader = DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=test_transformer),
                             batch_size=128, num_workers=workers,shuffle=False, pin_memory=True, drop_last=False)

    return [dataset, num_classes, train_loader, test_loader, init_loader, name]

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = transforms.Compose([
        transforms.Resize((height, width), interpolation=3),
        transforms.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def get_trans_train_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = transforms.Compose([
        transforms.Resize((height, width), interpolation=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop((height, width)),
        transforms.ToTensor(),
        # normalizer,
        # T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    test_transformer = transforms.Compose([
        transforms.Resize((height, width), interpolation=3),
        transforms.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def build_data_loaders(cfg, training_set, testing_only_set, toy_num=0):
    # Create data loaders
    data_dir = cfg.data_dir
    height, width = (256, 128)
    training_loaders = [get_data(name, data_dir, height, width, cfg.batch_size, cfg.workers,
                                 cfg.num_instances, select_num=500) for name in training_set]

  
    testing_loaders = [get_data(name, data_dir, height, width, cfg.batch_size, cfg.workers,
                                cfg.num_instances) for name in testing_only_set]
    return training_loaders, testing_loaders
