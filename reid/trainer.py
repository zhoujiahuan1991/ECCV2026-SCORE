from __future__ import print_function, absolute_import
import time

from torch.nn import functional as F
import torch
import torch.nn as nn

from reid.loss.loss_uncertrainty import TripletLoss_set
from .utils.meters import AverageMeter
from reid.metric_learning.distance import cosine_similarity

class Trainer(object):
    def __init__(self,args, model, recon_model, CC_flag, writer=None):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.recon_model = recon_model
        self.writer = writer
        self.uncertainty=True
        if self.uncertainty:
            self.criterion_ce=nn.CrossEntropyLoss()
            self.criterion_triple=TripletLoss_set()

        self.start_clust_epoch = args.start_clust_epoch
        
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        
        self.AF_weight=args.AF_weight   # anti-forgetting loss

        self.clo_ratio=args.clo_ratio   # anti-forgetting loss

        self.div1_weight=args.div1_weight
        self.div2_weight=args.div2_weight
        self.div3_weight=args.div3_weight
        self.div4_weight=args.div4_weight

        self.CC_flag = CC_flag
             
        self.n_sampling=args.n_sampling        

        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
       
    def train(self, epoch, data_loader_train, optimizer, optimizer_recon, training_phase,
              train_iters=200, add_num=0, old_model=None, proto_type=None        
              ):       
        self.model.train()
        # freeze the bn layer totally
        for m in self.model.module.base.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad == False and m.bias.requires_grad == False:
                    m.eval()
        if proto_type is not None:
            proto_type_merge={}
            steps=list(proto_type.keys())
            steps.sort()
            stages=1
            if stages<len(steps):
                steps=steps[-stages:]

            proto_type_merge['mean_features']=torch.cat([proto_type[k]['mean_features'] for k in steps])
            proto_type_merge['labels'] = torch.tensor([proto_type[k]['labels'] for k in steps]).to(proto_type_merge['mean_features'].device)
            proto_type_merge['mean_vars'] = torch.cat([proto_type[k]['mean_vars'] for k in steps])

            # proto_type_merge['clust_mean_features'] = torch.cat([proto_type[k]['clust_mean_features'] for k in steps])
            # proto_type_merge['clust_mean_vars'] = torch.cat([proto_type[k]['clust_mean_vars'] for k in steps])
           
            features_mean=proto_type_merge['mean_features']
            # clust_features_mean=proto_type_merge['clust_mean_features']

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_div1 = AverageMeter()
        losses_div2 = AverageMeter()
        losses_div3 = AverageMeter()
        losses_div4 = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            train_inputs = data_loader_train.next()
            ################################################################
            if train_inputs[0].size(0) != 128: #不满就取下一个
                train_inputs = data_loader_train.next()
            ################################################################
            data_time.update(time.time() - end)

            s_inputs, targets, cids, domains, = self._parse_data(train_inputs)
            targets += add_num
            # BS*2048, BS*(1+n_sampling)*2048,BS*(1+n_sampling)*num_classes, BS*2048
            # 先提取旧特征
            origin_feat_old = None
            if training_phase>1:
                with torch.no_grad():
                    features_old, merge_feat_old, _, origin_var_old, origin_feat_old = old_model(s_inputs, fkd=True)
            # 然后扔到新模型里提取特征
            s_features, merge_feat, cls_outputs, out_var, feat_final_layer  = self.model(s_inputs)

            loss_ce, loss_tp = 0, 0
            if self.uncertainty:
                ###ID loss###
                for s_id in range(1 + self.n_sampling):
                    loss_ce += self.criterion_ce(cls_outputs[:, s_id], targets)
                loss_ce = loss_ce / (1 + self.n_sampling)
                loss_ce=loss_ce*1
                ###set triplet-loss##
                loss_tp = self.criterion_triple(merge_feat, targets)[0]
                loss_tp=loss_tp*1.5            

            
            loss = loss_ce + loss_tp          

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tp.item())


            if training_phase>1:
                ##############################################################################################################################################
                # KL divergence
                divergence=0.                       
                # 样本级别affinity matrix
                center_feat=merge_feat[:,0]  # obtain the center feature
                Affinity_matrix_new = self.get_normal_affinity(center_feat, self.args.lambda_2) # obtain the affinity matrix

                features_var = proto_type_merge['mean_vars']    # obtain the prototype strandard variances
                noises = torch.randn(features_mean.size()).to(features_mean.device) # generate gaussian noise
                samples = noises * features_var + features_mean # obtain noised sample
                samples = F.normalize(samples, dim=1)   # normalize the sample

                s_features_old = cosine_similarity(center_feat, samples)    # obtain the new-old relation
                s_features_old = F.softmax(s_features_old /self.args.lambda_1, dim=1)  # normalize the relation

                Affinity_matrix_old = self.get_normal_affinity(s_features_old, self.args.lambda_2)  # obtain the affinity matrix under the prototype view
                # divergence += self.cal_KL(Affinity_matrix_new, Affinity_matrix_old, targets)
                Affinity_matrix_new_log = torch.log(Affinity_matrix_new)
                divergence+=self.KLDivLoss(Affinity_matrix_new_log, Affinity_matrix_old)
                ##############################################################################################################################################
                if epoch>=self.start_clust_epoch:
                    # -------------------------------------------------------------------------------------------------------------------------
                    # 想法1，选择性转移
                    # -------------------------------------------------------------------------------------------------------------------------
                    matrix_old = self.get_normal_affinity(features_old)
                    matrix_new = self.get_normal_affinity(s_features)
                    # 筛选条件
                    target_diff = targets.unsqueeze(0) != targets.unsqueeze(1)
                    domain_diff = domains.unsqueeze(0) != domains.unsqueeze(1)
                    domain_diff = domain_diff.to(matrix_old.device)
                    
                    # 初始化 rectified 矩阵
                    matrix_rectified = torch.zeros_like(matrix_old).to(matrix_old.device)

                    # 条件1：人不一样，保留旧的值
                    matrix_rectified[target_diff] = matrix_old[target_diff]

                    # 条件2：人一样，服装不一样，clo_ratio越大，旧的越多（因为旧的和新的主要是外观差异，所以多从旧的里面学习知识，以适应旧的）
                    matrix_rectified[~target_diff & domain_diff] = (
                        matrix_old[~target_diff & domain_diff] * self.clo_ratio +
                        matrix_new[~target_diff & domain_diff] * (1 - self.clo_ratio)
                    )

                    # 条件3：人一样，服装一样，clo_ratio越大，新的越多（因为旧的和新的主要是域差异，所以多从新的里面学习知识，以适应新的）
                    matrix_rectified[~target_diff & ~domain_diff] = (
                        matrix_old[~target_diff & ~domain_diff] * (1 - self.clo_ratio) +
                        matrix_new[~target_diff & ~domain_diff] * self.clo_ratio
                    )
                    
                    matrix_rectified = (matrix_rectified / matrix_rectified.sum(1, keepdim=True)).detach()
                    matrix_new_log = torch.log(matrix_new)
                    div1=self.KLDivLoss(matrix_new_log, matrix_rectified) * self.div1_weight
                    if self.CC_flag == False:
                        div1 *= 0.

                    loss = loss + div1 
                    losses_div1.update((div1).item())
                    # -------------------------------------------------------------------------------------------------------------------------
                    # 想法2，重构
                    # --------------------------------------------------------------------------------------
                    # Idea 2
                    matrix_old = self.get_normal_affinity(features_old)
                    # 重构旧特征
                    recon_input = F.normalize(s_features, p=2, dim=1)
                    # 如果少于20步，域映射网络不应该影响正常的学习
                    if epoch<self.start_clust_epoch:
                        recon_input = recon_input.detach()
                    recon_output = self.recon_model(recon_input)
                    # 重构后向原始特征对齐
                    matrix_new2old = self.get_normal_affinity(recon_output)
                    div2=self.KLDivLoss(torch.log(matrix_new2old), matrix_old.detach()) * self.div2_weight
                    if self.CC_flag == False:
                        div2 *= 0.

                    loss = loss + div2
                    losses_div2.update((div2).item())
                    # -------------------------------------------------------------------------------------------------------------------------
                    # 想法3+4，平衡蒸馏
                    # --------------------------------------------------------------------------------------
                    def merge_var(features_all, labels_all, var_all):
                        features_collect = {}
                        var_collect = {}
                        for feature, label, var in zip(features_all, labels_all,var_all):
                            label = label.cpu().item()
                            if label in features_collect:
                                features_collect[label].append(feature)
                                var_collect[label].append(var)
                            else:
                                features_collect[label] = [feature]
                                var_collect[label]=[var]
                        
                        # print (features_collect.keys())
                        labels_named = list(set(labels_all))
                        labels_named.sort()
                        # print (labels_named)
                        features_mean=[]
                        vars_mean=[]
                        for x in labels_named:
                            x = x.cpu().item()
                            if x in features_collect.keys():
                                feats=torch.stack(features_collect[x])
                                feat_mean=feats.mean(dim=0)
                                features_mean.append(feat_mean)

                                vars_2=(torch.stack(var_collect[x])**2).mean(dim=0)+(feats**2).mean(dim=0)-feat_mean**2
                                vars_mean.append(torch.sqrt(vars_2))
                            else:
                                features_mean.append(torch.zeros_like(features_all[0]))
                                vars_mean.append(torch.zeros_like(var_all[0]))
                        
                        return torch.stack(features_mean), torch.stack(vars_mean)
                    
                    fg_old_mean, fg_old_var = merge_var(features_old, domains, origin_var_old) # 新图旧模
                    fg_new_mean, fg_new_var = merge_var(s_features, domains, out_var) # 新图新模
                    
                    
                    def cal_aff(center_feat, center_feat_old):
                        ### 得到新模到旧模（待学习）
                        new2old_feat = cosine_similarity(center_feat, samples)
                        new2old_feat = F.softmax(new2old_feat / self.args.lambda_1, dim=1)
                        af_matrix_new = self.get_normal_affinity(new2old_feat, self.args.lambda_2)
                        af_matrix_new_log = torch.log(af_matrix_new)
                        ### 得到旧模到旧模（学习目标）
                        old2old_feat = cosine_similarity(center_feat_old, samples)
                        old2old_feat = F.softmax(old2old_feat / self.args.lambda_1, dim=1)
                        af_matrix_old = self.get_normal_affinity(old2old_feat, self.args.lambda_2)

                        return self.KLDivLoss(af_matrix_new_log, af_matrix_old)
                    
                    
                    ins_div = 0.
                    # 计算ins_div
                    ins_div += cal_aff(merge_feat[:,0], merge_feat_old[:,0])

                    fg_div = 0.
                    # 计算fg_div
                    fg_div += cal_aff(fg_new_mean, fg_old_mean)


                    # 融合
                    div3 = ins_div * self.div3_weight
                    if self.CC_flag == False:
                        div3 *= 0.

                    div4 = fg_div * self.div4_weight
                    if self.CC_flag == False:
                        div4 *= 0.
                    # ------------------------------------------------------------------------------------------------------
                    loss = loss + div3 + div4

                    losses_div3.update((div3).item())
                    losses_div4.update((div4).item())
                
                ##############################################################################################################################################
                loss = loss + divergence * self.AF_weight
                ##############################################################################################################################################
                
                
                

            optimizer.zero_grad()
            optimizer_recon.zero_grad()

            loss.backward()

            optimizer.step()
            optimizer_recon.step()
           
            batch_time.update(time.time() - end)
            end = time.time()
            if self.writer != None :
                self.writer.add_scalar(tag="loss/Loss_ce_{}".format(training_phase), scalar_value=losses_ce.val,
                          global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_tr_{}".format(training_phase), scalar_value=losses_tr.val,
                          global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_div1_{}".format(training_phase), scalar_value=losses_div1.val,
                          global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_div2_{}".format(training_phase), scalar_value=losses_div2.val,
                          global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_div3_{}".format(training_phase), scalar_value=losses_div3.val,
                          global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_div4_{}".format(training_phase), scalar_value=losses_div4.val,
                          global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="time/Time_{}".format(training_phase), scalar_value=batch_time.val,
                          global_step=epoch * train_iters + i)
            if (i + 1) == train_iters:
            #if 1 :
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tp {:.3f} ({:.3f})\t'                     
                      'Loss_div1 {:.3f} ({:.3f})\t'                     
                      'Loss_div2 {:.3f} ({:.3f})\t'    
                      'Loss_div3 {:.3f} ({:.3f})\t'                     
                      'Loss_div4 {:.3f} ({:.3f})\t'                     
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              losses_div1.val, losses_div1.avg,
                              losses_div2.val, losses_div2.avg,
                              losses_div3.val, losses_div3.avg,
                              losses_div4.val, losses_div4.avg,
                              ))
        


    def get_normal_affinity(self,x,Norm=0.1):
        pre_matrix_origin=cosine_similarity(x,x)
        pre_affinity_matrix=F.softmax(pre_matrix_origin/Norm, dim=1)
        return pre_affinity_matrix




    def _parse_data(self, inputs):
        imgs, _, pids, cids, domains = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets, cids, domains

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    # 多核MMD
    def mmd(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss

