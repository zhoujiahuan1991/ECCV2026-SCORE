from __future__ import print_function, absolute_import
import argparse
import copy
import os.path as osp
import sys

from torch.backends import cudnn
import torch.nn as nn
import random
from config import cfg
from reid.evaluators import Evaluator
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.feature_tools import *
from reid.models.layers import DataParallel
from reid.models.resnet_MM2 import make_model
from reid.trainer_zf1_new_aug_final_paper import Trainer
from torch.utils.tensorboard import SummaryWriter
from lreid_dataset.datasets.get_data_loaders_s_w_final_paper import build_data_loaders
from tools.Logger_results import Logger_res
import os


def apply_setting2_paper_defaults(args, argv):
    """Fill omitted setting-2 args with the paper's default recipe."""
    if args.setting != 2 or args.evaluate or args.test_folder:
        return args

    passed_flags = set()
    for token in argv[1:]:
        if token.startswith('--'):
            passed_flags.add(token.split('=')[0])
        elif token in ('-b', '-j'):
            passed_flags.add(token)

    default_values = {
        'seed': 0,
        'batch_size': 128,
        'workers': 4,
        'alpha': 0.5,
        'final_alpha': 0.5,
        'proto_pull_weight': 0.0,
        'proto_pull_tau': 0.5,
        'proto_pull_topk_weight': 0.5,
        'proto_pull_topk': 5,
    }
    flag_aliases = {
        'seed': ('--seed',),
        'batch_size': ('-b', '--batch-size'),
        'workers': ('-j', '--workers'),
        'alpha': ('--alpha',),
        'final_alpha': ('--final_alpha',),
        'proto_pull_weight': ('--proto_pull_weight',),
        'proto_pull_tau': ('--proto_pull_tau',),
        'proto_pull_topk_weight': ('--proto_pull_topk_weight',),
        'proto_pull_topk': ('--proto_pull_topk',),
    }

    for name, value in default_values.items():
        if not any(flag in passed_flags for flag in flag_aliases[name]):
            setattr(args, name, value)

    return args


def get_blend_alpha(args, stage_index, num_stages):
    """Use the tuned blend only after the final continual-learning stage."""
    return args.final_alpha if stage_index == num_stages - 1 else args.alpha


def combine_model_parameters(primary_model, reference_model, primary_weight):
    """Create a single model by uniformly averaging compatible parameters."""
    if not 0.0 <= primary_weight <= 1.0:
        raise ValueError('primary_weight must be in [0, 1]')

    combined_model = copy.deepcopy(primary_model)
    combined_state = combined_model.state_dict()
    primary_state = primary_model.state_dict()
    reference_state = reference_model.state_dict()

    for key, primary_value in primary_state.items():
        reference_value = reference_state.get(key)
        if reference_value is not None and reference_value.shape == primary_value.shape:
            combined_state[key] = (
                primary_weight * primary_value
                + (1.0 - primary_weight) * reference_value
            )

    combined_model.load_state_dict(combined_state)
    return combined_model


def load_blended_checkpoint_sequence(args, model, checkpoint_folder,
                                     checkpoint_names):
    """Load raw stage checkpoints and reproduce SCORE's recursive blending."""
    checkpoint = load_checkpoint(osp.join(checkpoint_folder, checkpoint_names[0]))
    copy_state_dict(checkpoint['state_dict'], model)

    for step, checkpoint_name in enumerate(checkpoint_names[1:], start=1):
        previous_model = copy.deepcopy(model)
        checkpoint = load_checkpoint(osp.join(checkpoint_folder, checkpoint_name))
        copy_state_dict(checkpoint['state_dict'], model)
        blend_alpha = get_blend_alpha(args, step, len(checkpoint_names))
        model = linear_combination(
            args,
            model,
            previous_model,
            blend_alpha,
        )

    return model

def main():
    args = parser.parse_args()
    args = apply_setting2_paper_defaults(args, sys.argv)

    if args.seed is not None: # 配合init_worker后可复现
        np.random.seed(args.seed) 
        random.seed(args.seed)
        
        os.environ['PYTHONHASHSEED'] = str(args.seed)  # 禁止hash随机化
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现
        
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed) # 多GPU训练需要设置这个
        torch.manual_seed(args.seed)
        
        torch.use_deterministic_algorithms(True) # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
        torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
        torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
        torch.backends.cudnn.benchmark = False  # 与上面一条代码配套使用，True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现。

    cfg.merge_from_file(args.config_file)
    main_worker(args, cfg)


def main_worker(args, cfg):
    log_name = 'log.txt'
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    else:
        log_dir = osp.dirname(args.test_folder)
        sys.stdout = Logger(osp.join(log_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))
    log_res_name='log_res.txt'
    logger_res=Logger_res(osp.join(args.logs_dir, log_res_name))    # record the test results
    

    """
    loading the datasets:
    setting： 1 or 2 
    """
    if 1 == args.setting:
        training_set = ['market1501', 'ltcc', 'prcc', 'msmt17', 'cuhk03']
        args.num_per_stage = [500, -1, -1, 500, 500]
        all_set = ['market1501', 'ltcc', 'prcc', 'msmt17', 'cuhk03',
                'ltcc_cc', 'prcc_cc']
    elif 2 == args.setting:
        training_set = ['msmt17', 'prcc', 'market1501', 'cuhk03', 'ltcc']
        args.num_per_stage = [500, -1, 500, 500, -1]
        all_set = ['market1501', 'prcc', 'msmt17', 'ltcc', 'cuhk03',
                'ltcc_cc', 'prcc_cc']
        
    elif 3 == args.setting:
        training_set = ['ltcc', 'cuhk03', 'prcc', 'msmt17', 'market1501']
        args.num_per_stage = [ -1,  500, -1, 500, 500]
        all_set = ['market1501', 'prcc', 'msmt17', 'ltcc', 'cuhk03',
                'ltcc_cc', 'prcc_cc']
    elif 4 == args.setting:
        training_set = ['market1501', 'prcc', 'cuhk03', 'ltcc', 'msmt17']
        args.num_per_stage = [500, -1, 500, -1, 500]
        all_set = ['market1501', 'prcc', 'msmt17', 'ltcc', 'cuhk03',
                'ltcc_cc', 'prcc_cc'] 
    elif 5 == args.setting:
        training_set = ['msmt17', 'ltcc', 'market1501', 'prcc', 'cuhk03']
        args.num_per_stage = [500, -1, 500, -1,500]
        all_set = ['market1501', 'prcc', 'msmt17', 'ltcc', 'cuhk03',
                'ltcc_cc', 'prcc_cc'] 
    elif 6 == args.setting:
        training_set = ['prcc', 'market1501', 'msmt17', 'ltcc', 'cuhk03']
        args.num_per_stage = [-1, 500, 500, -1, 500 ]
        all_set = ['market1501', 'prcc', 'msmt17', 'ltcc', 'cuhk03',
                'ltcc_cc', 'prcc_cc']
        
    # the datsets only used for testing
    testing_only_set = [x for x in all_set if x not in training_set]
    # get the loders of different datasets
    all_train_sets, all_test_only_sets = build_data_loaders(args, training_set, testing_only_set)    
    
    first_train_set = all_train_sets[0]
    model = make_model(args, num_class=first_train_set[1], camera_num=0, view_num=0, prototypes_per_class=args.prototypes_per_class)

    model.cuda()
    model = DataParallel(model)    
    writer = SummaryWriter(log_dir=args.logs_dir)
    # Load from checkpoint
    '''test the models under a folder'''
    if args.test_folder:
        ckpt_name = [x + '_checkpoint.pth.tar' for x in training_set]   # obatin pretrained model name
        checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[0]))  # load the first model
        copy_state_dict(checkpoint['state_dict'], model)     #    
        for step in range(len(ckpt_name) - 1):
            model_old = copy.deepcopy(model)    # backup the old model            
            checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[step + 1]))
            copy_state_dict(checkpoint['state_dict'], model)

            best_alpha = get_blend_alpha(args, step + 1, len(ckpt_name))
            model = linear_combination(args, model, model_old, best_alpha)

            save_name = '{}_checkpoint_adaptive_ema_{:.4f}.pth.tar'.format(training_set[step+1], best_alpha)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': 0,
                'mAP': 0,
            }, True, fpath=osp.join(args.logs_dir, save_name))
        if args.soup_folder:
            reference_model = load_blended_checkpoint_sequence(
                args,
                copy.deepcopy(model),
                args.soup_folder,
                ckpt_name,
            )
            model = combine_model_parameters(
                model,
                reference_model,
                primary_weight=args.soup_weight,
            )
            print(
                'Applied uniform model soup: primary={} reference={} '
                'primary_weight={:.3f}'.format(
                    args.test_folder,
                    args.soup_folder,
                    args.soup_weight,
                )
            )
        # extract_test_features(model, all_train_sets, all_test_only_sets, args.test_folder)
        test_model(model, all_train_sets, all_test_only_sets, len(all_train_sets)-1,logger_res=logger_res)

        exit(0)
    

    # resume from a model
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['mAP']
        print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))
   
    # Evaluator
    if args.MODEL in ['50x']:
        out_channel = 2048
    else:
        raise AssertionError(f"the model {args.MODEL} is not supported!")

    # train on the datasets squentially
    for set_index in range(0, len(training_set)):       
        model_old = copy.deepcopy(model)
        model = train_dataset(cfg, args, all_train_sets, all_test_only_sets, set_index, model, out_channel,
                                            writer,logger_res=logger_res)
        if set_index>0:

            best_alpha = get_blend_alpha(args, set_index, len(training_set))
            print(f'modified best_alpha: {best_alpha}')
            model = linear_combination(args, model, model_old, best_alpha)
        test_model(model, all_train_sets, all_test_only_sets, len(all_train_sets)-1,logger_res=logger_res)
  
    print('finished')

def get_normal_affinity(x,Norm=100):
    from reid.metric_learning.distance import cosine_similarity
    pre_matrix_origin=cosine_similarity(x,x)
    pre_affinity_matrix=F.softmax(pre_matrix_origin*Norm, dim=1)
    return pre_affinity_matrix
def get_adaptive_alpha(args, model, model_old, all_train_sets, set_index):
    dataset_new, num_classes_new, train_loader_new, _, init_loader_new, name_new = all_train_sets[
        set_index]  # trainloader of current dataset
    features_all_new, labels_all, fnames_all, camids_all, features_mean_new, labels_named = extract_features_voro(model,
                                                                                                          init_loader_new,
                                                                                                          get_mean_feature=True)
    features_all_old, _, _, _, features_mean_old, _ = extract_features_voro(model_old,init_loader_new,get_mean_feature=True)

    features_all_new=torch.stack(features_all_new, dim=0)
    features_all_old=torch.stack(features_all_old,dim=0)
    Affin_new = get_normal_affinity(features_all_new)
    Affin_old = get_normal_affinity(features_all_old)

    Difference= torch.abs(Affin_new-Affin_old).sum(-1).mean()

    alpha=float(1-Difference)
    return alpha



def train_dataset(cfg, args, all_train_sets, all_test_only_sets, set_index, model, out_channel, writer,logger_res=None ):
    dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[set_index]  # status of current dataset    

    print(f'===========Training on {name}, classes num: {num_classes}===========')
    Epochs= args.epochs0 if 0==set_index else args.epochs          

    if set_index == 0:
        #add_num = 0
        old_model=None
     
    else:
        '''store the old model'''
        old_model = copy.deepcopy(model)
        old_model = old_model.cuda()
        old_model.eval()

        model.module.classifier = nn.Linear(out_channel,num_classes, bias=False)
        model.cuda()

        classifier_centers, _, _ = mean_variance1(model, init_loader)
        model.module.classifier.weight.data.copy_(classifier_centers)
        model.cuda()

    classifier_centers, class_centers, class_vars = mean_variance1(model, init_loader)
    model.module.classifier.weight.data.copy_(classifier_centers)
    print(f'classifier_centers: {classifier_centers}')
    print(f'class_vars: {class_vars}')

    expanded_class_centers = classifier_centers.repeat_interleave(args.prototypes_per_class, dim=0)
    expandded_class_vars = class_vars.repeat_interleave(args.prototypes_per_class, dim=0)


    model.module.multiprototypes = nn.Parameter(torch.zeros(num_classes*args.prototypes_per_class, 2048))
    model.module.multivars = nn.Parameter(torch.ones(num_classes*args.prototypes_per_class, 2048))
    noise = torch.normal(mean=0, std=torch.sqrt(expandded_class_vars)) 
    #print(f"noise shape: {noise.shape}")
    expanded_class_centers_with_noise = expanded_class_centers + noise
    expanded_class_centers_with_noise = F.normalize(expanded_class_centers_with_noise, dim=1).float().cuda()
    model.module.multiprototypes.data.copy_(expanded_class_centers_with_noise)

    vars_noise = torch.randn(num_classes * args.prototypes_per_class, 2048) * 1e-6 * 5
    vars_noise = vars_noise.cuda()
    expandded_class_vars_with_noise = expandded_class_vars + vars_noise
    relu = nn.ReLU() 
    expandded_class_vars_with_noise = relu(expandded_class_vars_with_noise) + 1e-5
    
    
    model.module.multivars.data.copy_(expandded_class_vars_with_noise)
    model.cuda()

    # Re-initialize optimizer
    params = []
    for key, value in model.named_params(model):
        if not value.requires_grad:
            print('not requires_grad:', key)
            continue
        # else:
        #     print('requires_grad:', key)
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=args.momentum)    
    Stones=args.milestones
    lr_scheduler = WarmupMultiStepLR(optimizer, Stones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)
    
  
    trainer = Trainer(cfg, args, model, num_classes,  writer=writer)

    print('####### starting training on {} #######'.format(name))
    for epoch in range(0, Epochs):

        train_loader.new_epoch()
        trainer.train(epoch, train_loader,  optimizer, training_phase=set_index + 1,
                      train_iters=len(train_loader), old_model=old_model, num_classes=num_classes, prototypes_per_class=args.prototypes_per_class
                      )
        lr_scheduler.step()       
       

        if ((epoch + 1) % args.eval_epoch == 0 or epoch+1==Epochs):
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': 0.,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))

            logger_res.append('epoch: {}'.format(epoch + 1))
            
            mAP=0.
            if args.middle_test:
                test_model(model, all_train_sets, all_test_only_sets, len(all_train_sets)-1,logger_res=logger_res)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))  
        if epoch % 10 == 0:
            print(f'self.model.module.multiprototypes:{model.module.multiprototypes}')
            print(f'self.model.module.multivars:{model.module.multivars}')

    return model 

def test_model(model, all_train_sets, all_test_sets, set_index,  logger_res=None):
    begin = 0
    evaluator = Evaluator(model)
        
    R1_all = []
    mAP_all = []
    names=''
    Results=''
    train_mAP=0
    for i in range(begin, set_index + 1):
        dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[i]
        print('Results on {}'.format(name))
        if '_cc' in name:
            train_R1, train_mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True, change_clothes=True)  # ,training_phase=i+1)
        else:
            train_R1, train_mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True, change_clothes=False)  # ,training_phase=i+1)
        R1_all.append(train_R1)
        mAP_all.append(train_mAP)
        names = names + name + '\t\t'
        Results=Results+'|{:.1f}/{:.1f}\t'.format(train_mAP* 100, train_R1* 100)

    aver_mAP = torch.tensor(mAP_all).mean()
    aver_R1 = torch.tensor(R1_all).mean()


    R1_all = []
    mAP_all = []
    names_unseen = ''
    Results_unseen = ''
    for i in range(len(all_test_sets)):
        dataset, num_classes, train_loader, test_loader, init_loader, name = all_test_sets[i]
        print('Results on {}'.format(name))
        if '_cc' in name:
            R1, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True, change_clothes=True)
        else:
            R1, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True, change_clothes=False)

        R1_all.append(R1)
        mAP_all.append(mAP)
        names_unseen = names_unseen + name + '\t'
        Results_unseen = Results_unseen + '|{:.1f}/{:.1f}\t'.format(mAP* 100, R1* 100)

    aver_mAP_unseen = torch.tensor(mAP_all).mean()
    aver_R1_unseen = torch.tensor(R1_all).mean()

    print("Average mAP on Seen dataset: {:.1f}%".format(aver_mAP * 100))
    print("Average R1 on Seen dataset: {:.1f}%".format(aver_R1 * 100))
    names = names + '|Average\t|'
    Results = Results + '|{:.1f}/{:.1f}\t|'.format(aver_mAP * 100, aver_R1 * 100)
    print(names)
    print(Results)
    '''_________________________'''
    print("Average mAP on unSeen dataset: {:.1f}%".format(aver_mAP_unseen * 100))
    print("Average R1 on unSeen dataset: {:.1f}%".format(aver_R1_unseen * 100))
    names_unseen = names_unseen + '|Average\t|'
    Results_unseen = Results_unseen + '|{:.1f}/{:.1f}\t|'.format(aver_mAP_unseen* 100, aver_R1_unseen* 100)
    print(names_unseen)
    print(Results_unseen)
    if logger_res:
        logger_res.append(names)
        logger_res.append(Results)
        logger_res.append(Results.replace('|','').replace('/','\t'))
        logger_res.append(names_unseen)
        logger_res.append(Results_unseen)
        logger_res.append(Results_unseen.replace('|', '').replace('/', '\t'))
    return train_mAP



def linear_combination(args, model, model_old, alpha, model_old_id=-1):
    '''old model '''
    model_old_state_dict = model_old.state_dict()
    '''latest trained model'''
    model_state_dict = model.state_dict()

    ''''create new model'''
    model_new = copy.deepcopy(model)
    model_new_state_dict = model_new.state_dict()
    '''fuse the parameters'''
    for k, v in model_state_dict.items():
        if model_old_state_dict[k].shape == v.shape:
            # print(k,'+++')
                model_new_state_dict[k] = alpha * v + (1 - alpha) * model_old_state_dict[k]
        else:
            print(k, '...')
            # num_class_old = model_old_state_dict[k].shape[0]
            # model_new_state_dict[k][:num_class_old] = alpha * v[:num_class_old] + (1 - alpha) * model_old_state_dict[k]
    model_new.load_state_dict(model_new_state_dict)
    return model_new


def extract_centers(all_train_sets, set_index, model):
    dataset_old, num_classes_old, train_loader_old, _, init_loader_old, name_old = all_train_sets[set_index]
    features_all = []
    labels_all = []
    fnames_all = []
    camids_all = []

    model.train()
    with torch.no_grad():
        for i, (imgs, fnames, pids, cids, domains) in enumerate(init_loader_old):
            mean_feat, _, _,_ = model(imgs)
            for fname, feature, pid, cid in zip(fnames, mean_feat, pids, cids):
                features_all.append(feature)
                labels_all.append(int(pid))
                fnames_all.append(fname)
                camids_all.append(cid)


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
            feats=torch.stack(features_collect[x])
            feat_mean=feats.mean(dim=0)
            features_mean.append(feat_mean)
        else:
            features_mean.append(torch.zeros_like(features_all[0]))

    features_mean = torch.stack(features_mean)
    features_mean.requires_grad =False
    labels_named = torch.tensor(labels_named).to(features_mean.device)
    return  features_mean, labels_named

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model    
    parser.add_argument('--MODEL', type=str, default='50x',
                        choices=['50x'])
    # optimizer
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'],
                        help="optimizer ")
    parser.add_argument('--lr', type=float, default=0.008,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[30],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs0', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print-freq', type=int, default=200)
    
    # path   
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='../DATA/PRID')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join('../logs/try'))

    parser.add_argument('--config_file', type=str, default='config/base.yml',
                        help="config_file")
  
    parser.add_argument('--test_folder', type=str, default=None, help="test the models in a file")
    parser.add_argument('--soup_folder', type=str, default=None,
                        help="optional second checkpoint sequence for uniform model soup")
    parser.add_argument('--soup_weight', default=0.5, type=float,
                        help="weight of test_folder model in the model soup")
   
    parser.add_argument('--setting', type=int, default=1, help="training order setting")
    parser.add_argument('--middle_test', action='store_true', help="test during middle step")
    parser.add_argument('--AF_weight', default=1.0, type=float, help="anti-forgetting weight")   
    parser.add_argument('--alpha', default=0.5, type=float,
                        help="model fusion weight (new model) in non-final-stage linear_combination")
    parser.add_argument('--final_alpha', default=0.5, type=float, help="final-stage model blending weight")

    parser.add_argument('--prototypes_per_class', default=5, type=int)
    parser.add_argument('--save_evaluation', action='store_true', help="save ranking results")

    parser.add_argument('--learn_weight', default=1.0, type=float)
    parser.add_argument('--topk', default=10, type=int)
    parser.add_argument('--proto_pull_weight', default=0.0, type=float,
                        help='weight for pulling similar cross-identity prototypes together')
    parser.add_argument('--proto_pull_tau', default=0.5, type=float,
                        help='cosine similarity threshold for proto pull')
    parser.add_argument('--proto_pull_topk_weight', default=0.0, type=float,
                        help='weight for pulling each prototype top-k cross-identity neighbors')
    parser.add_argument('--proto_pull_topk', default=5, type=int,
                        help='top-k cross-identity neighbors used by proto pull top-k')
    
    parser.add_argument('--n_samples', default=5, type=int)
    parser.add_argument('--lambda_1', default=0.1, type=float, help="temperature")
    parser.add_argument('--lambda_2', default=0.1, type=float, help="temperature")

    main()
