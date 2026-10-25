import torch
import torch.nn.functional as F

from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from .data.sampler import RandomIdentitySampler, MultiDomainRandomIdentitySampler

import collections
import numpy as np
import copy

import random

def worker_init_fn(worked_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def extract_features(model, data_loader):
    features_all = []
    labels_all = []
    fnames_all = []
    camids_all = []
    normalized_features_all= []
    model.eval()
    with torch.no_grad():
        for i, (imgs, fnames, pids, cids, domains) in enumerate(data_loader):
            normalized_features, features = model(imgs, extract_centers=True)
            for fname, normalized_feature, feature, pid, cid in zip(fnames, normalized_features,features, pids, cids):
                normalized_features_all.append(normalized_feature)
                features_all.append(feature)
                labels_all.append(int(pid))
                fnames_all.append(fname)
                camids_all.append(cid)
    model.train()
    return normalized_features_all, features_all, labels_all, fnames_all, camids_all


# def initial_classifier(model, data_loader):
#     pid2features = collections.defaultdict(list)
#     features_all, labels_all, fnames_all, camids_all = extract_features(model, data_loader)
#     for feature, pid in zip(features_all, labels_all):
#         pid2features[pid].append(feature)
#     class_centers = [torch.stack(pid2features[pid]).mean(0) for pid in sorted(pid2features.keys())]
#     class_centers = torch.stack(class_centers)
#     return F.normalize(class_centers, dim=1).float().cuda()

def mean_variance(model, data_loader):
    pid2normalizedfeatures = collections.defaultdict(list)
    pid2features = collections.defaultdict(list)
    normalized_features_all, features_all, labels_all, fnames_all, camids_all = extract_features(model, data_loader)
    
    # 按照pid将特征分组
    for normalized_feature, feature, pid in zip(normalized_features_all, features_all, labels_all):
        pid2normalizedfeatures[pid].append(normalized_feature)
        pid2features[pid].append(feature)
    
    # 计算每个pid的特征中心和方差
    class_centers = []
    class_variances = []
    classifier_centers = []
    for pid in sorted(pid2features.keys()):
        normalized_features = torch.stack(pid2normalizedfeatures[pid])
        features = torch.stack(pid2features[pid])
        
        classifier_center = normalized_features.mean(0)
        center = features.mean(0)

        classifier_centers.append(classifier_center)
        class_centers.append(center)
        
        variance = features.var(0)  # 沿着特征维度计算方差
        class_variances.append(variance)
    classifier_centers = torch.stack(classifier_centers)
    class_centers = torch.stack(class_centers)
    class_variances = torch.stack(class_variances)

    return F.normalize(classifier_centers, dim=1).float().cuda(), class_centers.float().cuda(), class_variances.float().cuda()


def obtain_voronoi_loader(dataset,new_labels, add_num=0, batch_size = 32,num_instance=4,workers=8):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((256, 128)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    voronoi_set=[]

    if not isinstance(new_labels, list):
        new_labels = new_labels.tolist()

    for instance, lablel in zip(dataset.train, new_labels):    
        a=(instance[0], lablel, instance[2], instance[3])
        voronoi_set.append(a
            
        )     

    voronoi_loader = DataLoader(Preprocessor(voronoi_set, dataset.images_dir, train_transformer),
                             batch_size=batch_size,num_workers=workers, sampler=RandomIdentitySampler(voronoi_set, num_instance),
                             pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
    # new_labels.sort()

    return voronoi_loader, voronoi_set
def extract_features_uncertain(model, data_loader, get_mean_feature=False, eps=None, min_samples=None):
    features_all = []
    labels_all = []
    fnames_all = []
    camids_all = []
    var_all=[]
    model.train()
    with torch.no_grad():
        for i, (imgs, fnames, pids, cids, domains) in enumerate(data_loader):
            mean_feat, merge_feat, cls_outputs, out_var,_ = model(imgs)
            for fname, feature, pid, cid, var in zip(fnames, mean_feat, pids, cids,out_var):
                features_all.append(feature)
                labels_all.append(int(pid))
                fnames_all.append(fname)
                camids_all.append(cid)
                var_all.append(var)

    # features_all, labels_all, fnames_all, camids_all = extract_features(model, data_loader)

    if get_mean_feature:
        # 得ID的特征
        features_collect = {}
        var_collect = {}

        for feature, label, var in zip(features_all, labels_all,var_all):
            if label in features_collect:
                features_collect[label].append(feature)
                var_collect[label].append(var)
            else:
                features_collect[label] = [feature]
                var_collect[label]=[var]
        labels_named = list(set(labels_all))  # obtain valid features
        labels_named.sort()
        features_mean=[]
        vars_mean=[]
        for x in labels_named:
            if x in features_collect.keys():
                feats=torch.stack(features_collect[x])
                feat_mean=feats.mean(dim=0)
                features_mean.append(feat_mean)

                vars_2=(torch.stack(var_collect[x])**2).mean(dim=0)+(feats**2).mean(dim=0)-feat_mean**2
                vars_mean.append(torch.sqrt(vars_2))
            else:
                features_mean.append(torch.zeros_like(features_all[0]))
                vars_mean.append(torch.zeros_like(var_all[0]))
        ##############################################################################################################################################
        # # 得聚类的特征
        # ### 计算聚类标签
        # from sklearn.cluster import DBSCAN
        # cluster = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=8)
        # clust_labels_all = cluster.fit_predict(torch.stack(features_all).cpu())
        # ### 聚类标签
        # clust_features_collect = {}
        # clust_var_collect = {}
        # clust_id_collect = {}

        # for clust_feature, clust_label, person_label, clust_var in zip(features_all, clust_labels_all, labels_all, var_all):
        #     if clust_label in clust_features_collect:
        #         clust_features_collect[clust_label].append(clust_feature)
        #         clust_var_collect[clust_label].append(clust_var)
        #         clust_id_collect[clust_label].add(person_label)
        #     else:
        #         clust_features_collect[clust_label] = [clust_feature]
        #         clust_var_collect[clust_label]=[clust_var]
        #         clust_id_collect[clust_label]=set()
        #         clust_id_collect[clust_label].add(person_label)
        
        # clust_labels_named = list(set(clust_labels_all))  # 所有标签名
        # clust_labels_named.sort() # 标签名排序
        # clust_features_mean=[]
        # clust_vars_mean=[]

        # # 开始计算每个聚类结果
        # for clust_x in clust_labels_named:
        #     if (clust_x in clust_features_collect.keys()) and (clust_x != -1) and (len(clust_id_collect[clust_x]) >= 2): # 在采集的字典里，不是孤立点，有2个以上编号
        #         clust_feats=torch.stack(clust_features_collect[clust_x])
        #         clust_feat_mean=clust_feats.mean(dim=0)
        #         clust_features_mean.append(clust_feat_mean)

        #         clust_vars_2=(torch.stack(clust_var_collect[clust_x])**2).mean(dim=0)+(clust_feats**2).mean(dim=0)-clust_feat_mean**2
        #         clust_vars_mean.append(torch.sqrt(clust_vars_2))
        #     # else:
        #     #     clust_features_mean.append(torch.zeros_like(features_all[0]))
        #     #     clust_vars_mean.append(torch.zeros_like(var_all[0]))

        # return features_all, labels_all, fnames_all, camids_all, torch.stack(features_mean),labels_named,torch.stack(vars_mean),var_all, torch.stack(clust_features_mean), torch.stack(clust_vars_mean)
        ##############################################################################################################################################
        return features_all, labels_all, fnames_all, camids_all, torch.stack(features_mean),labels_named,torch.stack(vars_mean),var_all
    else:
        return features_all, labels_all, fnames_all, camids_all
def extract_features_voro(model, data_loader, get_mean_feature=False):    

    features_all, labels_all, fnames_all, camids_all = extract_features(model, data_loader)

    if get_mean_feature:
        features_collect = {}        

        for feature, label in zip(features_all, labels_all):
            if label in features_collect:
                features_collect[label].append(feature)               
            else:
                features_collect[label] = [feature]                
        labels_named = list(set(labels_all))  # obtain valid features
        labels_named.sort()
        features_mean=[]        
        for x in labels_named:
            if x in features_collect.keys():
                features_mean.append(torch.stack(features_collect[x]).mean(dim=0))                
            else:
                features_mean.append(torch.zeros_like(features_all[0]))
                
        return features_all, labels_all, fnames_all, camids_all, torch.stack(features_mean),labels_named
    else:
        return features_all, labels_all, fnames_all, camids_all



def select_replay_samples(model, dataset, training_phase=0, add_num=0, old_datas=None, select_samples=2,batch_size = 32,workers=8):
    replay_data = []
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    train_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((256, 128)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_loader = DataLoader(Preprocessor(dataset.train, root=dataset.images_dir,transform=transformer),
                              batch_size=batch_size, num_workers=workers, shuffle=True, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

    features_all, labels_all, fnames_all, camids_all = extract_features(model, train_loader)

    pid2features = collections.defaultdict(list)
    pid2fnames = collections.defaultdict(list)
    pid2cids = collections.defaultdict(list)

    for feature, pid, fname, cid in zip(features_all, labels_all, fnames_all, camids_all):
        pid2features[pid].append(feature)
        pid2fnames[pid].append(fname)
        pid2cids[pid].append(cid)

    labels_all = list(set(labels_all))

    class_centers = [torch.stack(pid2features[pid]).mean(0) for pid in sorted(pid2features.keys())]
    class_centers = F.normalize(torch.stack(class_centers), dim=1)
    select_pids = np.random.choice(labels_all, 250, replace=False)
    for pid in select_pids:
        feautures_single_pid = F.normalize(torch.stack(pid2features[pid]), dim=1, p=2)
        center_single_pid = class_centers[pid]
        simi = torch.mm(feautures_single_pid, center_single_pid.unsqueeze(0).t())
        simi_sort_inx = torch.sort(simi, dim=0)[1][:2]
        for id in simi_sort_inx:
            replay_data.append((pid2fnames[pid][id], pid+add_num, pid2cids[pid][id], training_phase-1))

    if old_datas is None:
        data_loader_replay = DataLoader(Preprocessor(replay_data, dataset.images_dir, train_transformer),
                             batch_size=batch_size,num_workers=workers, sampler=RandomIdentitySampler(replay_data, select_samples),
                             pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
    else:
        replay_data.extend(old_datas)
        data_loader_replay = DataLoader(Preprocessor(replay_data, dataset.images_dir, train_transformer),
                             batch_size=training_phase*batch_size,num_workers=workers,
                             sampler=MultiDomainRandomIdentitySampler(replay_data, select_samples),
                             pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)

    return data_loader_replay, replay_data


def get_pseudo_features(data_specific_batch_norm, training_phase, x, domain, unchange=False):
    fake_feat_list = []
    if unchange is False:
        for i in range(training_phase):
            if int(domain[0]) == i:
                data_specific_batch_norm[i].train()
                fake_feat_list.append(data_specific_batch_norm[i](x)[..., 0, 0])
            else:
                data_specific_batch_norm[i].eval()
                fake_feat_list.append(data_specific_batch_norm[i](x)[..., 0, 0])
                data_specific_batch_norm[i].train()
    else:
        for i in range(training_phase):
            data_specific_batch_norm[i].eval()
            fake_feat_list.append(data_specific_batch_norm[i](x)[..., 0, 0])

    return fake_feat_list
