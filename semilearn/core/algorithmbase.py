# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# from gc import freeze
import os
import contextlib
import numpy as np
from inspect import signature
from collections import OrderedDict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from semilearn.core.hooks import Hook, get_priority, CheckpointHook, TimerHook, LoggingHook, DistSamplerSeedHook, ParamUpdateHook, EvaluationHook, EMAHook, WANDBHook, AimHook
from semilearn.core.utils import get_dataset, get_data_loader, get_optimizer, get_cosine_schedule_with_warmup, Bn_Controller
from semilearn.core.criterions import CELoss, ConsistencyLoss

from scipy.optimize import linear_sum_assignment
import copy
import torch_clustering
# class NoisyLabel(nn.Module):
#     def __init__(self, num_classes,mom=0.999,n=6000):
#         super(NoisyLabel, self).__init__()
#         self.mom=mom
#         self.n=n
#         self.classes=num_classes
#         self.logit_bank = torch.zeros(self.n, self.classes).cuda()
#         self.mark=torch.ones(self.n).cuda()
#     def update(self,logit,label,idx):
#         self.logit_bank[idx] = self.mom * self.logit_bank[idx] + (1 - self.mom) * logit
#         _,pre=torch.max(self.logit_bank[idx],dim=1)
#         self.mark[idx]= torch.min(self.mark[idx],(pre == label).float())
#         #self.mark[idx]= (pre == label).float()
#     def getmark(self,idx):
#         return self.mark[idx]
#     def getmoveinglabel(self):
#         return torch.max(self.logit_bank,dim=1)[1].cpu().numpy()



class AlgorithmBase:
    """
        Base class for algorithms
        init algorithm specific parameters and common parameters
        
        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
    """
    def __init__(
        self,
        args,
        net_builder,
        tb_log=None,
        logger=None,
        **kwargs):
        
        # common arguments
        self.match=None
        self.args = args
        self.num_classes = args.num_classes
        self.ema_m = args.ema_m
        self.epochs = args.epoch
        self.num_train_iter = args.num_train_iter
        self.num_eval_iter = args.num_eval_iter
        self.num_log_iter = args.num_log_iter
        self.num_iter_per_epoch = int(self.num_train_iter // self.epochs)
        self.lambda_u = args.ulb_loss_ratio 
        self.use_cat = args.use_cat
        self.use_amp = args.amp
        self.clip_grad = args.clip_grad
        self.save_name = args.save_name
        
        self.save_dir = args.save_dir
        self.resume = args.resume
        self.algorithm = args.algorithm
        self.use_ema_for_train = args.use_ema_for_train
        self.cluster_pseudo_label_weight = args.cluster_weight
        #self.num_train_lb_update_iter = args.num_train_lb_update_iter
        # commaon utils arguments
        self.tb_log = tb_log
        self.print_fn = print if logger is None else logger.info
        self.print_fn(self.save_name)
        self.ngpus_per_node = torch.cuda.device_count()
        self.loss_scaler = GradScaler()
        self.amp_cm = autocast if self.use_amp else contextlib.nullcontext
        self.gpu = args.gpu
        self.rank = args.rank
        self.distributed = args.distributed
        self.world_size = args.world_size
        
        # common model related parameters
        self.it = 0
        self.start_epoch = 0
        self.best_eval_acc, self.best_it = 0.0, 0
        self.bn_controller = Bn_Controller()
        self.net_builder = net_builder
        self.ema = None
        self.test_index=None
        # build dataset
        self.dataset_dict = self.set_dataset()
        # self.noisylabel=NoisyLabel(self.num_classes,0.5,self.args.lb_dest_len)
        # build data loader
        self.loader_dict = self.set_data_loader()

        #self.cluster_weight = cluster_weight

        # cv, nlp, speech builder different arguments
        self.model = self.set_model()
        #checkpoint = torch.load(args.load_path, map_location='cpu')
        #print(self.model.load_state_dict(checkpoint['model'],strict=False))
        

        # build optimizer and scheduler
        self.optimizer, self.scheduler = self.set_optimizer()

        # build supervised loss and unsupervised loss
        self.ce_loss = CELoss()
        self.consistency_loss = ConsistencyLoss()

        # other arguments specific to the algorithm
        # self.init(**kwargs)

        # set common hooks during training
        self._hooks = []  # record underlying hooks 
        self.hooks_dict = OrderedDict() # actual object to be used to call hooks
        self.set_hooks()
        base_dir=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        dump_dir = os.path.join(base_dir, 'data', self.args.dataset, 'labeled_idx')
        os.makedirs(dump_dir, exist_ok=True)
        lb_dump_path = os.path.join(dump_dir, self.args.label_idx_name)
        self.lb_idx = np.load(lb_dump_path)
        self.lb_targets = None
        self.freezen=False
        self.cluster_label_for_train=None
        self.center_feature=None
        self.norm_center_feature_for_ema=None
        self.center_feature_for_ema=None
        self.best_cluster_eval={'acc':0.0}
        self.best_head_eval={'acc':0.0}
        

    def init(self, **kwargs):
        """
        algorithm specific init function, to add parameters into class
        """
        raise NotImplementedError
    

    def set_dataset(self):
        """
        set dataset_dict
        """
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        dataset_dict = get_dataset(self.args, self.algorithm, self.args.dataset, self.args.num_labels, self.args.num_classes, self.args.data_dir, self.args.include_lb_to_ulb)
        if dataset_dict is None:
            return dataset_dict

        self.args.ulb_dest_len = len(dataset_dict['train_ulb']) if dataset_dict['train_ulb'] is not None else 0
        self.args.lb_dest_len = len(dataset_dict['train_lb'])
        
        self.print_fn("unlabeled data number: {}, labeled data number {}".format(self.args.ulb_dest_len, self.args.lb_dest_len))
        if self.rank == 0 and self.distributed:
            torch.distributed.barrier()
        return dataset_dict

    def set_data_loader(self):
        """
        set loader_dict
        """
        if self.dataset_dict is None:
            return
            
        self.print_fn("Create train and test data loaders")
        loader_dict = {}
        loader_dict['train_lb'] = get_data_loader(self.args,
                                                  self.dataset_dict['train_lb'],
                                                  self.args.batch_size,
                                                  data_sampler=self.args.train_sampler,
                                                  num_iters=self.num_train_iter,
                                                  num_epochs=self.epochs,
                                                  num_workers=self.args.num_workers,
                                                  distributed=self.distributed)
        

        loader_dict['train_ulb'] = get_data_loader(self.args,
                                                   self.dataset_dict['train_ulb'],
                                                   self.args.batch_size * self.args.uratio,
                                                   data_sampler=self.args.train_sampler,
                                                   num_iters=self.num_train_iter,
                                                   num_epochs=self.epochs,
                                                   num_workers=2 * self.args.num_workers,
                                                   distributed=self.distributed)

        loader_dict['eval'] = get_data_loader(self.args,
                                              self.dataset_dict['eval'],
                                              self.args.eval_batch_size,
                                              # make sure data_sampler is None for evaluation
                                              data_sampler=None,
                                              num_workers=self.args.num_workers,
                                              drop_last=False)
        
        if self.dataset_dict['test'] is not None:
            loader_dict['test'] =  get_data_loader(self.args,
                                                   self.dataset_dict['test'],
                                                   self.args.eval_batch_size,
                                                   # make sure data_sampler is None for evaluation
                                                   data_sampler=None,
                                                   num_workers=self.args.num_workers,
                                                   drop_last=False)
        self.print_fn(f'[!] data loader keys: {loader_dict.keys()}')
        return loader_dict

    def set_optimizer(self):
        """
        set optimizer for algorithm
        """
        self.print_fn("Create optimizer and scheduler")
        self.print_fn(self.args.num_warmup_iter)
        optimizer = get_optimizer(self.model, self.args.optim, self.args.fast_lr,self.args.slow_lr, self.args.momentum, self.args.weight_decay, self.args.layer_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    self.num_train_iter,
                                                    num_warmup_steps=self.args.num_warmup_iter)
        return optimizer, scheduler

    def set_model(self):
        """
        initialize model
        """
        model = self.net_builder(num_classes=self.num_classes, pretrained=self.args.use_pretrain, pretrained_path=self.args.pretrain_path)
        return model

    def set_ema_model(self):
        """
        initialize ema model from model
        """
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def set_hooks(self):
        """
        register necessary training hooks
        """
        # parameter update hook is called inside each train_step
        self.register_hook(ParamUpdateHook(), None, "HIGHEST")
        self.register_hook(EMAHook(), None, "HIGH")
        self.register_hook(EvaluationHook(), None, "HIGH")
        self.register_hook(CheckpointHook(), None, "HIGH")
        self.register_hook(DistSamplerSeedHook(), None, "NORMAL")
        self.register_hook(TimerHook(), None, "LOW")
        self.register_hook(LoggingHook(), None, "LOWEST")
        if self.args.use_wandb:
            self.register_hook(WANDBHook(), None, "LOWEST")
        if self.args.use_aim:
            self.register_hook(AimHook(), None, "LOWEST")

    

    def process_batch(self, input_args=None, **kwargs):
        """
        process batch data, send data to cuda
        NOTE **kwargs should have the same arguments to train_step function as keys to work properly
        """
        if input_args is None:
            input_args = signature(self.train_step).parameters
            input_args = list(input_args.keys())

        input_dict = {}
        
        for arg, var in kwargs.items():
            if not arg in input_args:
                continue
            
            if var is None:
                continue
            
            # send var to cuda
            if isinstance(var, dict):
                var = {k: v.cuda(self.gpu) for k, v in var.items()}
            else:
                var = var.cuda(self.gpu)

            #if arg=='y_lb':
            #    reordered_preds = torch.zeros(var.shape[0],dtype=var.dtype).cuda(self.gpu)
            #    for pred_i, target_i in self.match:
            #        reordered_preds[var == int(target_i)] = int(pred_i)
            #    var=reordered_preds
            input_dict[arg] = var
            
        return input_dict
    

    def process_out_dict(self, out_dict=None, **kwargs):
        """
        process the out_dict as return of train_step
        """
        if out_dict is None:
            out_dict = {}

        for arg, var in kwargs.items():
            out_dict[arg] = var
        
        # process res_dict, add output from res_dict to out_dict if necessary
        return out_dict


    def process_log_dict(self, log_dict=None, prefix='train', **kwargs):
        """
        process the tb_dict as return of train_step
        """
        if log_dict is None:
            log_dict = {}

        for arg, var in kwargs.items():
            log_dict[f'{prefix}/' + arg] = var
        return log_dict

    def compute_prob(self, logits):
        return torch.softmax(logits, dim=-1)

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        """
        train_step specific to each algorithm
        """
        # implement train step for each algorithm
        # compute loss
        # update model 
        # record log_dict
        # return log_dict
        raise NotImplementedError

    
        
   
    
    
    def train(self):
        """
        train function
        """
        
        self.model.train()
        #self.ema_model = self.set_ema_model()
        self.call_hook("before_run")
        #self.test_index=self.get_label_idx()
        
        self.print_fn(self.evaluate(change=True))
        #k=1
        
        #self.print_fn(self.evaluate())
        
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break
            
            self.call_hook("before_train_epoch")

            if self.freezen:
                self.freeze_all(self.model.module.encoder_k,False)
                self.print_fn("freeze backbone  ! ! !")
            for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                         self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break
                '''
                if self.it<=self.args.num_warmup_iter:
                    self.cluster_pseudo_label_weight = 0.0
                else:
                    self.cluster_pseudo_label_weight = self.cluster_weight
                '''
                
                self.call_hook("before_train_step")
                

                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                

                #if (self.it+1)>4096 and (self.it+1) %  self.num_train_lb_update_iter==0:
                #    self.update_lbdataset(1)
                
                self.call_hook("after_train_step")
                
                if self.freezen and (self.it+1)>self.args.num_eval_iter*10:
                    #self.reload_lr(self.args.lr)
                    self.freeze_all(self.model.module.encoder_k,True) 
                    self.model.train()
                    self.freezen=False
                    self.print_fn("start train all ! ! !")
                self.it += 1
                
                
            if self.freezen:
                self.freeze_all(self.model.module.encoder_k,True) 
                self.print_fn("defreeze backbone ! ! !")
            
            self.call_hook("after_train_epoch")

        self.call_hook("after_run")
        
    
    
    def _hungarian_match(self,flat_preds, flat_targets, preds_k, targets_k):
        # Based on implementation from IIC
        num_samples = flat_targets.shape[0]

        assert (preds_k == targets_k)  # one to one
        num_k = preds_k
        num_correct = np.zeros((num_k, num_k))

        for c1 in range(num_k):
            for c2 in range(num_k):
                # elementwise, so each sample contributes once
                votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
                num_correct[c1, c2] = votes

    # num_correct is small
        match = linear_sum_assignment(num_samples - num_correct)
        match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
        res = []
        for out_c, gt_c in match:
            res.append((out_c, gt_c))

        return res
    
    def pairwise_cosine(self,x1: torch.Tensor, x2: torch.Tensor, pairwise=True):
        #x1 = F.normalize(x1)
        #x2 = F.normalize(x2)
        if not pairwise:
            return (1 - (x1 * x2).sum(dim=1))
        return 1 - x1.mm(x2.T)

    def predict(self,X: torch.Tensor, cluster_centers_=None,need_norm=True):
        
        X= F.normalize(torch.tensor(X).cuda())
        if need_norm:
            cluster_centers_=F.normalize(torch.tensor(cluster_centers_).cuda())
        split_size = min(4096, X.size(0))
        inertia, pred_labels = 0., []
        for f in X.split(split_size, dim=0):
            d = self.pairwise_cosine(f,cluster_centers_)
            inertia_, labels_ = d.min(dim=1)
            inertia += inertia_.sum()
            pred_labels.append(d)
        return torch.cat(pred_labels, dim=0), inertia

    def clustering(self, features):
        features=torch.tensor(features).cuda()
        #self.print_fn(features)
        kwargs = {
            'metric': 'cosine',
            'distributed': True,
            'random_state': 0,
            'n_clusters': self.num_classes,
            'verbose': True
        }
        clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)

        psedo_labels = clustering_model.fit_predict(features)
        cluster_centers = clustering_model.cluster_centers_
        self.center_feature=cluster_centers
        
        #print()
        return psedo_labels,cluster_centers

    def simple_clustering_evaluate(self,y_true,y_pred):
        

        match = self._hungarian_match(torch.tensor(y_pred).cuda(self.gpu), torch.tensor(y_true).cuda(self.gpu), preds_k=self.num_classes, targets_k=self.num_classes)

        reordered_preds = np.zeros(y_true.shape[0],dtype=y_true.dtype)
        for pred_i, target_i in match:
            reordered_preds[y_pred == int(pred_i)] = int(target_i)
        self.print_fn(match)
        y_pred=reordered_preds
        top1 = accuracy_score(y_true, y_pred)
        nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
        ari = metrics.adjusted_rand_score(y_true, y_pred)
        return {'acc':top1,'nmi':nmi,'ari':ari},y_pred

    def clustering_evaluate(self,y_true,y_pred):
        

        match = self._hungarian_match(torch.tensor(y_pred).cuda(self.gpu), torch.tensor(y_true).cuda(self.gpu), preds_k=self.num_classes, targets_k=self.num_classes)

        reordered_preds = np.zeros(y_true.shape[0],dtype=y_true.dtype)
        for pred_i, target_i in match:
            reordered_preds[y_pred == int(pred_i)] = int(target_i)
        y_pred=reordered_preds
        
        # np.save('img10_propos_label',np.array(y_pred))

        center_feature_tmp=torch.zeros_like(self.center_feature,device=self.center_feature.device)
        for pred_i, target_i in match:
            center_feature_tmp[target_i]=self.center_feature[pred_i]
        self.center_feature=center_feature_tmp
        #self.print_fn(match)
        top1 = accuracy_score(y_true, y_pred)
        nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
        ari = metrics.adjusted_rand_score(y_true, y_pred)
        return {'acc':top1,'nmi':nmi,'ari':ari},y_pred
    
    def get_labeled_index(self,p,features,psedo_labels,cluster_centers):
        clustering_model = torch_clustering.PyTorchKMeans(init='k-means++')
        dis=clustering_model.distance_metric(features,cluster_centers)
        all_var=dis.min(dim=1)[0]
        # all_var=dis.var(dim=1)

        pre_class=dis.shape[0]//self.num_classes
        labeled_count=int(pre_class*p)
        print(labeled_count)
        labeded_index=list()
        labeled_target=list()
        ori_index=torch.tensor(range(dis.shape[0])).cuda()
        for u in range(self.num_classes):
            # print(all_var)
            pos=torch.where(psedo_labels==u)[0]
            sub_var,sub_idx=(all_var[pos]).sort()
            sub_index=ori_index[pos][sub_idx]
            #print(sub_var)
            #print(all_var[sub_index])
            # tmpimdex=sub_index[:labeled_count]
            # indices = torch.randperm(len(sub_index))[:labeled_count]
            # tmpimdex=sub_index[indices]
            tmpimdex=sub_index[:labeled_count]
            # np.save('lb_index/'+self.opt.dataset+'_disditc',np.array(sub_var.detach().cpu().numpy()))
            labeded_index+=tmpimdex.cpu().numpy().tolist()
            labeled_target+=psedo_labels[tmpimdex].cpu().numpy().tolist()
        return np.array(labeded_index),np.array(labeled_target)
        

    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False,change=False):
        """
        evaluation function
        """
        from tqdm import tqdm
        self.ema_model.eval()
        self.model.eval()
        self.ema.apply_shadow()
        # if change:
        #     self.model.module.cluster_head_new.weight = torch.nn.Parameter(F.normalize(self.model.module.center_0,dim=0).t())
        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        # y_probs = []
        y_logits = []
        #index=[]
        features = []
        y_mask = []
        with torch.no_grad():
            
            for data in tqdm(eval_loader):
                x = data['x_lb']
                y = data['y_lb']
                #idx=data['idx_lb']
                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch
                tmp=self.model.forward(x,False)
                logits = tmp[out_key]
                #self.print_fn(self.model[0])
                loss = F.cross_entropy(logits, y, reduction='mean', ignore_index=-1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                if change==False:
                    y_mask.extend(self.call_hook("get_positive_mask", "MaskingHook",logits).cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                features.append(tmp['feat'].cpu().numpy())
                #index.extend(idx.cpu().tolist())
                total_loss += loss.item() * num_batch
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_mask = np.array(y_mask)
        features = np.concatenate(features)
        y_logits = np.concatenate(y_logits)
        cluster_labels=self.clustering(features)[0].cpu().numpy()

        #self.print_fn("feature cluster eval2 "+str(torch_clustering.evaluate_clustering(y_true,cluster_labels)))
        #self.print_fn("y_logits eval "+str(torch_clustering.evaluate_clustering(y_true,y_pred)))
        
        #index = np.array(index)
        match = self._hungarian_match(torch.tensor(y_pred).cuda(self.gpu), torch.tensor(y_true).cuda(self.gpu), preds_k=self.num_classes, targets_k=self.num_classes)
        self.print_fn(match)
        nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
        ari = metrics.adjusted_rand_score(y_true, y_pred)
        self.match=match
        reordered_preds = np.zeros(y_true.shape[0],dtype=y_true.dtype)
        for pred_i, target_i in self.match:
            reordered_preds[y_pred == int(pred_i)] = int(target_i)
        ori_pred=y_pred
        y_pred=reordered_preds

        msg,cluster_y_pred=self.clustering_evaluate(y_true,cluster_labels)
        if msg['acc']>self.best_cluster_eval['acc']:
            self.best_cluster_eval=msg
        
        
        # dis=y_logits[np.array([4588,5108,4091,4140,4060,4777,4758,4836,4748,4823,5149,4088,4067,5147])]
        # dis=torch.tensor(dis).softmax(dim=1)
        # for i in range(14):
        #     u=dis[i]
        #     reordered_preds = torch.zeros(dis[0].shape[0])
        #     for pred_i, target_i in self.match:
        #         reordered_preds[target_i]=u[pred_i]
        #     dis[i]=reordered_preds
        # print(dis)
        # dis=dis.numpy()
        # with open('/data/lizihan/new_v2.txt', 'a') as f:
        #     f.write("\n\n")
        #     # np.savetxt(f, dis.cpu().numpy(), delimiter=',', fmt='%.6f')
        #     np.savetxt(f, dis, delimiter=',', fmt='%.6f')
        # dis=self.predict(torch.tensor(features)[torch.tensor([4588,5108,4091,4140,4060,4777,4758,4836,4748,4823,5149,4088,4067,5147])].cuda(),self.center_feature,True)[0]
        # # print(self.center_feature.shape)
        # # print(torch.tensor(features[5108]).shape)
        # # dis=self.pairwise_cosine()
        # for i in range(14):
        #     dis[i]=dis[i]/dis[i].sum()
        
        # self.print_fn(dis)
        
        
        # 5108
        if 1:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            dump_dir = os.path.join(base_dir, 'data', self.args.dataset, 'labeled_idx')
            lb_dump_path = os.path.join(dump_dir, self.args.label_target_name)
            targets=torch.tensor(np.load(lb_dump_path).astype('int32')).cuda()
            match_2 = self._hungarian_match(torch.tensor(cluster_y_pred[self.lb_idx]).cuda(self.gpu),targets, preds_k=self.num_classes, targets_k=self.num_classes)
        else:
            match_2 = self._hungarian_match(torch.tensor(cluster_y_pred).cuda(self.gpu), torch.tensor(ori_pred).cuda(self.gpu), preds_k=self.num_classes, targets_k=self.num_classes)
        self.print_fn('cluster match:'+str(match_2))
        reordered_preds = np.zeros(y_true.shape[0],dtype=y_true.dtype)
        for pred_i, target_i in match_2:
            reordered_preds[cluster_y_pred == int(pred_i)] = int(target_i)
        self.cluster_label_for_train=torch.tensor(reordered_preds).cuda(self.gpu)

        # p=0.1
        if 0:
            p = 0.1
            labeded_index,labeled_target=self.get_labeled_index(0.1,torch.tensor(features).cuda(),self.cluster_label_for_train,self.center_feature)

            labeled_acc=accuracy_score(y_true[labeded_index], cluster_y_pred[labeded_index])
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            dump_dir = os.path.join(base_dir, 'data', self.args.dataset, 'labeled_idx')
            np.save(dump_dir+'/secu_'+str(p)+'_'+str(labeled_acc)+'_idx_'+str(len(labeded_index)),np.array(labeded_index))
            np.save(dump_dir+'/secu_'+str(p)+'_'+str(labeled_acc)+'_target_'+str(len(labeded_index)),np.array(labeled_target))
            self.print_fn('/secu_'+str(p)+'_'+str(labeled_acc)+'_target_'+str(len(labeded_index)))
            

        #self.print_fn(self.cluster_label_for_train)
        self.print_fn("feature cluster eval "+str(msg)) 
        self.print_fn("equal num ratio "+str((reordered_preds==ori_pred).sum()/y_pred.shape[0]))

        # if change==False:
        #     self.print_fn("kmeans right num ratio "+str(((cluster_y_pred==y_true)*y_mask).sum()/y_mask.sum()))
        #     self.print_fn("head   right num ratio "+str(((y_pred==y_true)*y_mask).sum()/y_mask.sum()))
        #     self.print_fn("select num ratio "+str(y_mask.sum()/len(y_mask)))

        

        


        # y_logits = np.concatenate(y_logits)
        top1 = accuracy_score(y_true, y_pred)
        balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')

        if top1>self.best_head_eval['acc']:
            self.best_head_eval={'acc':top1,'nmi':nmi,'ari':ari}
        self.print_fn("best feature eval "+str(self.best_cluster_eval))
        self.print_fn("best head eval "+str(self.best_head_eval))

        #test_index=self.test_index
        #test_cleanlb_acc = accuracy_score(y_true[test_index], y_pred[test_index])
        #self.print_fn("test_cleanlb_acc "+str(test_cleanlb_acc))
        lb_count = [0 for _ in range(self.num_classes)]
        for c in y_pred[self.lb_idx]:
            lb_count[c] += 1
        self.print_fn("pre lb count: {}".format(lb_count))
        
        if change:
            
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            dump_dir = os.path.join(base_dir, 'data', self.args.dataset, 'labeled_idx')
            lb_dump_path = os.path.join(dump_dir, self.args.label_target_name)
            
            targets=torch.tensor(np.load(lb_dump_path).astype('int32')).cuda()
            # match_tmp = self._hungarian_match(torch.tensor(ori_pred[self.lb_idx]).cuda(self.gpu),targets, preds_k=self.num_classes, targets_k=self.num_classes)
            # reordered_preds = torch.zeros(targets.shape[0]).cuda()
            # for pred_i, target_i in match_tmp:
            #     reordered_preds[targets == int(target_i)] = int(pred_i)
                
            targets=targets.detach().cpu().numpy().astype('int32')

            # targets=np.load(lb_dump_path).astype('int32')
            #self.print_fn(targets)
            self.dataset_dict['train_lb'].update_target(targets)
            self.lb_targets=targets
            lb_count = [0 for _ in range(self.num_classes)]
            for c in targets:
                lb_count[c] += 1
            self.print_fn("right lb count: {}".format(lb_count))
            
            #self.dataset_dict['train_lb'].update_target(ori_pred[self.lb_idx])
            # print(self.dataset_dict['train_lb'])
            self.loader_dict['train_lb'] = get_data_loader(self.args,
                                                  self.dataset_dict['train_lb'],
                                                  self.args.batch_size,
                                                  data_sampler=self.args.train_sampler,
                                                  num_iters=self.num_train_iter,
                                                  num_epochs=self.epochs,
                                                  num_workers=self.args.num_workers,
                                                  distributed=self.distributed)
            self.loader_dict['ori_train_lb'] =  get_data_loader(self.args,
                                                  copy.deepcopy(self.dataset_dict['train_lb']),
                                                  self.args.eval_batch_size,
                                                  data_sampler=None,
                                                  num_workers=self.args.num_workers,
                                                  drop_last=False)
            self.dataset_dict['train_lb'].init_ori_data()
            self.print_fn("change Sucess")
        
        
        # cleanlb_acc = accuracy_score(y_true[self.lb_idx], y_pred[self.lb_idx],sample_weight=self.noisylabel.mark.cpu().numpy())
        cleanlb_acc = accuracy_score(y_true[self.lb_idx], y_pred[self.lb_idx])
        
        
            
        
        # if not change:
        #     reordered_preds=np.zeros(self.lb_targets.shape[0],dtype=y_true.dtype)
        #     for pred_i, target_i in self.match:
        #         reordered_preds[self.lb_targets == int(pred_i)] = int(target_i)
        #     lb_pre_target=reordered_preds
        #     lb_right_ratio=accuracy_score(y_true[self.lb_idx], lb_pre_target,sample_weight=self.noisylabel.mark.cpu().numpy())
        #     if (1-self.noisylabel.mark).cpu().numpy().sum()>0.5:
        #         lb_error_right_ratio = accuracy_score(y_true[self.lb_idx], lb_pre_target,sample_weight=(1-self.noisylabel.mark).cpu().numpy())


        #cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        #self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        self.ema_model.train()
        

        eval_dict = {eval_dest+'/loss': total_loss / total_num, eval_dest+'/top-1-acc': top1, eval_dest+'/NMI': nmi,eval_dest+'/ARI': ari,
                    #  eval_dest+'/clean_label_ratio':self.noisylabel.mark.mean().item(),eval_dest+'/noisyacc':noisylb_acc,
                     eval_dest+'/cleanacc':cleanlb_acc,
                    #  eval_dest+'/lb_right_ratio':lb_right_ratio,eval_dest+'/lb_error_right_ratio':lb_error_right_ratio,
                     eval_dest+'/balanced_acc': balanced_top1, eval_dest+'/precision': precision, eval_dest+'/recall': recall, eval_dest+'/F1': F1}
        if return_logits:
            eval_dict[eval_dest+'/logits'] = y_logits
        #self.print_fn('eval_top1_ans:' + str(top1))
        return eval_dict

    
    def get_save_dict(self):
        """
        make easier for saving model when need save additional arguments
        """
        # base arguments for all models
        save_dict = {
            'model': self.model.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_scaler': self.loss_scaler.state_dict(),
            'it': self.it + 1,
            'epoch': self.epoch + 1,
            'best_it': self.best_it,
            'best_eval_acc': self.best_eval_acc,
        }
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        return save_dict
    

    def save_model(self, save_name, save_path):
        """
        save model and specified parameters for resume
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        save_filename = os.path.join(save_path, save_name)
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_filename)
        self.print_fn(f"model saved: {save_filename}")


    def load_model(self, load_path):
        """
        load model and specified parameters for resume
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        # self.loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        # self.it = checkpoint['it']
        # self.start_epoch = checkpoint['epoch']
        # self.epoch = self.start_epoch
        # self.best_it = checkpoint['best_it']
        # self.best_eval_acc = checkpoint['best_eval_acc']
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        # if self.scheduler is not None and 'scheduler' in checkpoint:
        #     self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.print_fn('Model loaded')
        return checkpoint

    def check_prefix_state_dict(self, state_dict):
        """
        remove prefix state dict in ema model
        """
        new_state_dict = dict()
        for key, item in state_dict.items():
            if key.startswith('module'):
                new_key = '.'.join(key.split('.')[1:])
            else:
                new_key = key
            new_state_dict[new_key] = item
        return new_state_dict

    def register_hook(self, hook, name=None, priority='NORMAL'):
        """
        Ref: https://github.com/open-mmlab/mmcv/blob/a08517790d26f8761910cac47ce8098faac7b627/mmcv/runner/base_runner.py#L263
        Register a hook into the hook list.
        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            hook_name (:str, default to None): Name of the hook to be registered. Default is the hook class name.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority  # type: ignore
        hook.name = name if name is not None else type(hook).__name__

        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:  # type: ignore
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        
        if not inserted:
            self._hooks.insert(0, hook)

        # call set hooks
        self.hooks_dict = OrderedDict()
        for hook in self._hooks:
            self.hooks_dict[hook.name] = hook
        


    def call_hook(self, fn_name, hook_name=None, *args, **kwargs):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            hook_name (str): The specific hook name to be called, such as
                "param_update" or "dist_align", uesed to call single hook in train_step.
        """
        
        if hook_name is not None:
            return getattr(self.hooks_dict[hook_name], fn_name)(self, *args, **kwargs)
        
        for hook in self.hooks_dict.values():
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self, *args, **kwargs)

    def registered_hook(self, hook_name):
        """
        Check if a hook is registered
        """
        return hook_name in self.hooks_dict


    @staticmethod
    def get_argument():
        """
        Get specificed arguments into argparse for each algorithm
        """
        return {}



class ImbAlgorithmBase(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        
        # imbalanced arguments
        self.lb_imb_ratio = self.args.lb_imb_ratio
        self.ulb_imb_ratio = self.args.ulb_imb_ratio
        self.imb_algorithm = self.args.imb_algorithm
    
    def imb_init(self, *args, **kwargs):
        """
        intiialize imbalanced algorithm parameters
        """
        pass 

    def set_optimizer(self):
        if 'vit' in self.args.net and self.args.dataset in ['cifar100', 'food101', 'semi_aves', 'semi_aves_out']:
            return super().set_optimizer() 
        elif self.args.dataset in ['imagenet', 'imagenet127']:
            return super().set_optimizer() 
        else:
            self.print_fn("Create optimizer and scheduler")
            optimizer = get_optimizer(self.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay, self.args.layer_decay, bn_wd_skip=False)
            scheduler = None
            return optimizer, scheduler
