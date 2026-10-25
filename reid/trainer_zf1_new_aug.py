from __future__ import print_function, absolute_import
import time

from torch.nn import functional as F
import torch
import torch.nn as nn
from .utils.meters import AverageMeter
from .utils.feature_tools import *

from reid.utils.make_loss import make_loss
import copy
from torch.distributions.multivariate_normal import MultivariateNormal
from reid.metric_learning.distance import cosine_similarity
class Trainer(object):
    def __init__(self,cfg,args, model, num_classes, writer=None):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.args = args
        self.model = model
        self.writer = writer
        self.AF_weight = args.AF_weight

        self.loss_fn, self.center_criterion = make_loss(cfg, num_classes=num_classes)
        self.BCEloss = nn.BCEWithLogitsLoss()
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    def train(self, epoch, data_loader_train,  optimizer, training_phase,
              train_iters=200, old_model=None, num_classes=0, prototypes_per_class=0
              ):

        self.model.train()
        # freeze the bn layer totally
        for m in self.model.module.base.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad == False and m.bias.requires_grad == False:
                    m.eval()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_ce_strong = AverageMeter()
        losses_tr_strong = AverageMeter()
        
        pro_losses_ce = AverageMeter()
      
        pro_losses_l2 = AverageMeter()
        prototype_losses_ce = AverageMeter()
        prototype_losses_tr = AverageMeter()
        prototype_losses_kl = AverageMeter()
        #prototype_losses_center = AverageMeter()

        pro_losses_kl = AverageMeter()
        losses_div = AverageMeter()
        end = time.time()
        prototype_labels = torch.arange(num_classes).repeat_interleave(prototypes_per_class).cuda()
        prototype_indexs = torch.arange(num_classes*prototypes_per_class).cuda()
        #print(prototype_labels)
        for i in range(train_iters):
            train_inputs = data_loader_train.next()
            data_time.update(time.time() - end)

            s_inputs, s_inputs_strong, targets, cids, domains = self._parse_data(train_inputs)
            batch_size = s_inputs.size(0)

            inputs_all = torch.cat((s_inputs, s_inputs_strong), dim=0)
            s_features_all, bn_feat, cls_outputs_all, feat_final_layer_all = self.model(inputs_all)
            s_features = s_features_all[:batch_size]
            s_features_strong = s_features_all[batch_size:]
            cls_outputs = cls_outputs_all[:batch_size]
            cls_outputs_strong = cls_outputs_all[batch_size:]
            if isinstance(feat_final_layer_all, torch.Tensor):
                feat_final_layer = feat_final_layer_all[:batch_size]
            else:
                feat_final_layer = feat_final_layer_all

            '''calculate the base loss'''
            loss_ce, loss_tp = self.loss_fn(cls_outputs, s_features, targets, target_cam=None)
            loss_ce_s, loss_tp_s = self.loss_fn(cls_outputs_strong, s_features_strong, targets, target_cam=None)
            
            loss = loss_ce + loss_tp + loss_ce_s + loss_tp_s
            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tp.item())
            losses_ce_strong.update(loss_ce_s.item())
            losses_tr_strong.update(loss_tp_s.item())


            #L_p START==============================================
            targets_list = targets.tolist() 
            unique_targets = list(dict.fromkeys(targets_list))  # [1, 3]
            unique_targets = torch.tensor(unique_targets).cuda()

            mask = torch.isin(prototype_labels, unique_targets)
            indices = torch.nonzero(mask, as_tuple=True)[0]
            selected_labels = prototype_labels[indices]
            selected_prototypes = self.model.module.multiprototypes[indices]
            selected_prototypes_BN = self.model.module.bottleneck(selected_prototypes.unsqueeze(2).unsqueeze(3))
            selected_cls = self.model.module.classifier(selected_prototypes_BN[..., 0, 0])

            prototype_loss_ce, prototype_loss_tr = self.loss_fn(selected_cls, selected_prototypes, selected_labels, target_cam=None)

            prototype_losses_ce.update(prototype_loss_ce.item())
            prototype_losses_tr.update(prototype_loss_tr.item())

            loss = loss + prototype_loss_ce + prototype_loss_tr
            #L_p END==============================================

            #L_BSE START==============================================
            dist1=self.euclidean_squared_distance(s_features, self.model.module.multiprototypes)# 得到bs x n_prototype的矩阵

            _, pro_indices = torch.topk(-dist1, self.args.topk, dim=1)
            selected_pro = self.model.module.multiprototypes[pro_indices]
            selected_pro_vars = self.model.module.multivars[pro_indices]
            selected_pro_labels = prototype_labels[pro_indices]
            selected_pro_index = prototype_indexs[pro_indices]


            
            pro_cls, pro_indices_max = self.cal_topk_pdf(s_features, selected_pro, selected_pro_vars)
            pro_labels = selected_pro_labels[torch.arange(pro_cls.size(0)), pro_indices_max]
            pro_indexs = selected_pro_index[torch.arange(pro_cls.size(0)), pro_indices_max]

            binary_labels = (targets == pro_labels).float()
            indices_of_correct = (binary_labels == 1).nonzero(as_tuple=True)[0]

            correct_match_indices = pro_indexs[indices_of_correct]
            #correct_match_indices = torch.unique(correct_match_indices)
            pro_loss_ce =self.BCEloss(pro_cls, binary_labels)

            if i == train_iters-1:
                print(f'binary_labels: {binary_labels}')
                print(correct_match_indices)
                print(f'correct_num:{len(correct_match_indices)}')
                print(pro_loss_ce)
            

            pro_loss_ce = torch.clamp(pro_loss_ce, min=0, max=5) 
            pro_losses_ce.update(pro_loss_ce.item())
            #L_BSE END==============================================

            #L_MSE START==============================================
            prototypes_per_class = self.args.prototypes_per_class  # 每个类别有多少原型
            start_indices = targets * prototypes_per_class  # [batch_size]

            offset = torch.arange(prototypes_per_class).to(targets.device)  # [prototypes_per_class]
            prototype_indices = start_indices.unsqueeze(1) + offset.unsqueeze(0)  # [batch_size, prototypes_per_class]

            selected_prototypes1 = self.model.module.multiprototypes[prototype_indices]  # [batch_size, prototypes_per_class, dim]
            selected_pro_vars1 = self.model.module.multiprototypes[prototype_indices]
            _, pro_indices_max1 = self.cal_topk_pdf(s_features, selected_prototypes1, selected_pro_vars1)

            best_prototypes = selected_prototypes1[torch.arange(pro_indices_max1.size(0)), pro_indices_max1]

            
            pro_loss_l2 = F.mse_loss(s_features, best_prototypes) 
            loss =  loss + (pro_loss_l2 + pro_loss_ce)*self.args.learn_weight
            #L_MSE END==============================================

            pro_losses_l2.update(pro_loss_l2.item())
            if i == train_iters-1:
                print(pro_loss_l2)
            
            if old_model is not None:
                old_model.eval()
                with torch.no_grad():
                    s_features_old, bn_feat_old, cls_outputs_old, feat_final_layer_old = old_model(s_inputs, get_all_feat=True)
                if isinstance(s_features_old, tuple):
                    s_features_old=s_features_old[0]
                Affinity_matrix_new = self.get_normal_affinity(s_features)
                Affinity_matrix_old = self.get_normal_affinity(s_features_old)
                divergence = self.cal_KL(Affinity_matrix_new, Affinity_matrix_old)
                losses_div.update(divergence.item())
                
                

                old_multivars = old_model.module.multivars
                old_multimeans =  old_model.module.multiprototypes

                _, dim = old_multimeans.shape
                old_expanded_means = old_multimeans.unsqueeze(1).expand(-1, self.args.n_samples, -1)
                old_expanded_stds = torch.sqrt(old_multivars).unsqueeze(1).expand(-1, self.args.n_samples, -1)
                old_noise = torch.normal(mean=0.0, std=old_expanded_stds)
                old_samples = old_expanded_means + old_noise 
                old_samples = old_samples.reshape(-1, dim)

                new_multivars = self.model.module.multivars[correct_match_indices]
                new_multimeans =  self.model.module.multiprototypes[correct_match_indices]

                new_expanded_means = new_multimeans.unsqueeze(1).expand(-1, self.args.n_samples, -1)
                new_expanded_stds = torch.sqrt(new_multivars).unsqueeze(1).expand(-1, self.args.n_samples, -1)
                new_noise = torch.normal(mean=0.0, std=new_expanded_stds)
                new_samples = new_expanded_means + new_noise 
                new_samples = new_samples.reshape(-1, dim)

                pro_sim_old = cosine_similarity(new_samples, old_samples)
                pro_sim_old = F.softmax(pro_sim_old / self.args.lambda_1, dim=1)

                # pro_sim_new = cosine_similarity(new_samples, new_samples)
                # pro_sim_new = F.softmax(pro_sim_new, dim=1)


                # pro_sim_new = cosine_similarity(new_samples, new_samples)
                # pro_sim_new = F.softmax(pro_sim_new, dim=1)
                # pro_Affinity_matrix_new = self.get_normal_affinity(pro_sim_new, self.args.lambda_2)


                pro_Affinity_matrix_old = self.get_normal_affinity(pro_sim_old, self.args.lambda_2)
                pro_Affinity_matrix_new = self.get_normal_affinity(new_samples, self.args.lambda_2)
                pro_loss_kl =  self.cal_KL(pro_Affinity_matrix_new, pro_Affinity_matrix_old)

                pro_losses_kl.update(pro_loss_kl.item())


                if i == train_iters-1:
                    print(f'proto_losses_kl: {pro_loss_kl}')
                loss = loss + (pro_loss_kl + divergence )*self.args.AF_weight



            optimizer.zero_grad()
            loss.backward()

            optimizer.step()           

            batch_time.update(time.time() - end)
            end = time.time()
            if self.writer != None :
                self.writer.add_scalar(tag="loss/Loss_ce_{}".format(training_phase), scalar_value=losses_ce.val,
                          global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_tr_{}".format(training_phase), scalar_value=losses_tr.val,
                          global_step=epoch * train_iters + i)

                self.writer.add_scalar(tag="time/Time_{}".format(training_phase), scalar_value=batch_time.val,
                          global_step=epoch * train_iters + i)
            if (i + 1) == train_iters:
            #if 1 :
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tp {:.3f} ({:.3f})\t'
                      'Loss_div {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              losses_div.val, losses_div.avg
                  ))       
                print(
                      'Loss_ce_strong {:.3f} ({:.3f})\t'
                      'Loss_tp_strong {:.3f} ({:.3f})\t'
                      .format(
                              losses_ce_strong.val, losses_ce_strong.avg,
                              losses_tr_strong.val, losses_tr_strong.avg,
                              )
                )

                print(
                      'pro_losses_ce {:.3f} ({:.3f})\t'
                      'pro_losses_kl {:.3f} ({:.3f})\t'
                      'prototype_losses_ce {:.3f} ({:.3f})\t'
                      'prototype_losses_tr {:.3f} ({:.3f})\t'
                      'prototype_losses_kl {:.3f} ({:.3f})\t'
                      .format(
                              pro_losses_ce.val, pro_losses_ce.avg,
                              pro_losses_kl.val, pro_losses_kl.avg,
                              prototype_losses_ce.val, prototype_losses_ce.avg,
                              prototype_losses_tr.val, prototype_losses_tr.avg,
                              prototype_losses_kl.val, prototype_losses_kl.avg,
                              )
                )

    def get_normal_affinity(self,x,Norm=0.1):
        pre_matrix_origin=cosine_similarity(x,x)
        pre_affinity_matrix=F.softmax(pre_matrix_origin/Norm, dim=1)
        return pre_affinity_matrix
    def _parse_data(self, inputs):
        imgs, imgs_strong, _, pids, cids, domains = inputs
        inputs = imgs.cuda()
        inputs_strong = imgs_strong.cuda()
        targets = pids.cuda()
        return inputs, inputs_strong, targets, cids, domains
    
    def cal_topk_pdf(self, X, mean_matrix, cov_diag_matrix):
        X_expanded = X.unsqueeze(1)  # 形状变为 (bs, 1, d)

        diff = X_expanded - mean_matrix  # 形状为 (bs, topk, d)
        #print(f'diff{diff}')

        # 计算指数部分 (x - μ)^2 / (2 * σ^2)
        exponent = 0.5* torch.sum((diff ** 2) / ( cov_diag_matrix), dim=2)  # 形状为 (bs, n_prototypes)
        # print(exponent.shape)
        # exit(0)
        eps = 1e-1
        coefficient = -1/2 * torch.log(2 * torch.pi * (cov_diag_matrix)).sum(dim=2)   # 形状为 (bs, n_prototypes)
        # coefficient = -d/2*torch.log(2 * torch.pi)-1/2*torch.log(cov_diag_expanded).sum(dim=2)   # 形状为 (1, n_prototypes)

        pdf_values = coefficient - exponent
        pdf_values_max, indices = pdf_values.max(dim=1)
       # print(f'indices: {indices}')
        return pdf_values_max, indices

    def euclidean_squared_distance(self, input1, input2):
        """Computes euclidean squared distance.
        Args:
            input1 (torch.Tensor): 2-D feature matrix.
            input2 (torch.Tensor): 2-D feature matrix.
        Returns:
            torch.Tensor: distance matrix.
        """
        m, n = input1.size(0), input2.size(0)
        distmat = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, input1, input2.t())
        return distmat
    
    def cal_selected_pdf(self, X, mean_matrix, cov_diag_matrix, targets):


        prototypes_per_class = self.args.prototypes_per_class  # 每个类别有多少原型
        start_indices = targets * prototypes_per_class  # [batch_size]

        batch_size = targets.size(0)

        offset = torch.arange(prototypes_per_class).to(targets.device)  # [prototypes_per_class]
        prototype_indices = start_indices.unsqueeze(1) + offset.unsqueeze(0)  # [batch_size, prototypes_per_class]
        # print(prototype_indices)
        # exit(0)
        selected_prototypes_mean = mean_matrix[prototype_indices]  # [batch_size, prototypes_per_class, dim]
        selected_prototypes_var = cov_diag_matrix[prototype_indices]
        sample_features_expanded = X.unsqueeze(1)  # [batch_size, 1, dim]

        diff = sample_features_expanded - selected_prototypes_mean  # 形状为 (bs, n_prototypes, d)

        exponent = 0.5* torch.sum((diff ** 2) / ( selected_prototypes_var), dim=2)  # 形状为 (n_samples, n_prototypes)
        eps = 1e-1
        coefficient = -1/2 * torch.log(2 * torch.pi * (selected_prototypes_var)).sum(dim=2)   # 形状为 (bs, n_prototypes)

        pdf_values = coefficient - exponent
        scale_factor = 5 / pdf_values.abs().max().item()
        pdf_values = pdf_values * scale_factor
        pdf_values_max = pdf_values.max(dim=1).values

        return pdf_values_max
    
    def cal_KL(self,Affinity_matrix_new, Affinity_matrix_old):
        Affinity_matrix_new_log = torch.log(Affinity_matrix_new)
        divergence=self.KLDivLoss(Affinity_matrix_new_log, Affinity_matrix_old) 
        return divergence 

    def get_attri(self, Gts, pre_affinity_matrix,margin=0):
        Thres_P=((1-Gts)*pre_affinity_matrix).max(dim=1,keepdim=True)[0]
        T_scores=pre_affinity_matrix*Gts

        TP=((T_scores-Thres_P)>margin).float()
        TP=torch.maximum(TP, torch.eye(TP.size(0)).to(TP.device))

        FN=Gts-TP

        Mapped_affinity=(1-Gts) +pre_affinity_matrix
        Mapped_affinity = Mapped_affinity+torch.eye(Mapped_affinity.size(0)).to(Mapped_affinity.device)
        Thres_N = Mapped_affinity.min(dim=1, keepdim=True)[0]
        N_scores=pre_affinity_matrix*(1-Gts)

        FP=(N_scores>Thres_N ).float()
        TN=(1-Gts) -FP
        attris={
            'TP':TP,
            'FN':FN,
            'FP':FP,
            'TN':TN,
            "Thres_P":Thres_P,
            "Thres_N":Thres_N
        }
        return attris

