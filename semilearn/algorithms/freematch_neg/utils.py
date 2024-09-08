# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from numbers import Rational
import torch

from semilearn.algorithms.utils import concat_all_gather
from semilearn.algorithms.hooks import MaskingHook






class FreeMatchThresholingHook(MaskingHook):
    """
    SAT in FreeMatch
    """
    def __init__(self, num_classes, momentum=0.999,topk=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.m = momentum
        
        self.p_model = torch.ones((self.num_classes)) / self.num_classes
        self.p_model_2 = torch.ones((self.num_classes)) / self.num_classes
        self.label_hist = torch.ones((self.num_classes)) / self.num_classes
        self.time_p = self.p_model.mean()
        self.all_tau=(torch.ones((self.num_classes,self.num_classes)) / self.num_classes).scatter_(1,torch.tensor(range(self.num_classes)).unsqueeze(dim=-1),0.0)
        self.tau=self.all_tau.sum(dim=1)
        # self.tau = torch.zeros((self.num_classes))
        self.tau_half=self.all_tau.sum(dim=1)
        self.topk=topk
    
    @torch.no_grad()
    def updata_neg2(self,algorithm,probs_x_ulb):
        #probs_w = torch.softmax(pred_w.detach(), dim=-1)
        val,ind = probs_x_ulb.topk(self.topk, 1, True, True)
        val=val.t()[self.topk-1]
        #negtar=ind.t()[1]
        target=ind.t()[0]
        
        negative_feature=(probs_x_ulb.clone().scatter_(1,target.unsqueeze(dim=-1),0.0))
        negative_feature=negative_feature/negative_feature.sum()
        self.all_tau= self.all_tau*self.m + (1 - self.m)*(negative_feature.mean(dim=0))
        self.tau=self.tau*self.m+(1-self.m)*val.mean()
    
    @torch.no_grad()
    def updata_neg2(self,algorithm,probs_x_ulb):
        #probs_w = torch.softmax(pred_w.detach(), dim=-1)
        val, target = probs_x_ulb.min(dim=-1)
        negative_feature=(probs_x_ulb.clone().scatter_(1,target.unsqueeze(dim=-1),0.0))

        cnt=torch.bincount(target,minlength=self.num_classes).clamp(1e-8)
        #tau_update = torch.bincount(target,weights=1-val,minlength=self.num_classes)/cnt
        all_tau_update=torch.zeros((self.num_classes,self.num_classes)).cuda().index_add_(0,target,negative_feature)/cnt.unsqueeze(dim=1)
        tau_update = torch.bincount(target,weights=val,minlength=self.num_classes)/cnt
        
        #negative_feature=negative_feature/negative_feature.sum()
        self.all_tau= self.all_tau*self.m + (1 - self.m)*all_tau_update
        self.tau=self.tau*self.m+(1-self.m)*tau_update

        


    @torch.no_grad()
    def getnegmask2(self,algorithm,logits_x_ulb):
        if not self.all_tau.is_cuda:
            self.all_tau = self.all_tau.to(logits_x_ulb.device)
        if not self.tau.is_cuda:
            self.tau = self.tau.to(logits_x_ulb.device)
        probs=torch.softmax(logits_x_ulb.detach(), dim=-1)
        self.updata_neg2(algorithm,probs)
        _, target = torch.max(logits_x_ulb, dim=-1)
        #nowthe=self.all_tau / torch.max(self.all_tau, dim=-1)[0] *self.tau
        #import pdb
        #pdb.set_trace()
        nowthe=self.all_tau/self.all_tau.min(dim=1)[0].clamp(1e-8).unsqueeze(dim=1) * self.tau.unsqueeze(dim=1)
        

        mask=probs<nowthe[target]
        mask=mask.clone().scatter_(1,target.t().unsqueeze(dim=-1),0).to(nowthe)
        self.neg_th=nowthe
        return mask

    @torch.no_grad()
    def updata_neg(self,algorithm,probs_x_ulb):
        #probs_w = torch.softmax(pred_w.detach(), dim=-1)
        val, target = probs_x_ulb.max(dim=-1)
        negative_feature=(probs_x_ulb.clone().scatter_(1,target.unsqueeze(dim=-1),0.0))

        cnt=torch.bincount(target,minlength=self.num_classes).clamp(1e-8)
        #tau_update = torch.bincount(target,weights=1-val,minlength=self.num_classes)/cnt
        all_tau_update=torch.zeros((self.num_classes,self.num_classes)).cuda().index_add_(0,target,negative_feature)/cnt.unsqueeze(dim=1)

        tau_update=all_tau_update.sum(dim=1)
        
        #negative_feature=negative_feature/negative_feature.sum()
        self.all_tau= self.all_tau*self.m + (1 - self.m)*all_tau_update
        self.tau=self.tau*self.m+(1-self.m)*tau_update

        '''
        tau_half_index = torch.nonzero(val>=self.tau[target]).squeeze(dim=1)
        val_half,target_half=val[tau_half_index], target[tau_half_index]
        
        cnt_half=torch.bincount(target_half,minlength=self.num_classes).clamp(1e-8)
        tau_half_update = torch.bincount(target_half,weights=1-val_half,minlength=self.num_classes)/cnt_half
        self.tau_half=self.tau_half*self.m+(1-self.m)*tau_half_update
        '''


    @torch.no_grad()
    def getnegmask(self,algorithm,logits_x_ulb):
        if not self.all_tau.is_cuda:
            self.all_tau = self.all_tau.to(logits_x_ulb.device)
        if not self.tau.is_cuda:
            self.tau = self.tau.to(logits_x_ulb.device)
        if not self.tau_half.is_cuda:
            self.tau_half = self.tau_half.to(logits_x_ulb.device)
        probs=torch.softmax(logits_x_ulb.detach(), dim=-1)
        self.updata_neg(algorithm,probs)
        _, target = torch.max(logits_x_ulb, dim=-1)
        #nowthe=self.all_tau / torch.max(self.all_tau, dim=-1)[0] *self.tau
        nowthe=self.all_tau/self.all_tau.sum(dim=1).clamp(1e-8).unsqueeze(dim=1) * self.tau.unsqueeze(dim=1)
        
        

        mask=probs<nowthe[target]
        # mask=probs<0.01
        mask=mask.clone().scatter_(1,target.t().unsqueeze(dim=-1),0).to(nowthe)
        self.neg_th=nowthe
        return mask
    
    @torch.no_grad()
    def update(self, algorithm, probs_x_ulb):
        if algorithm.distributed and algorithm.world_size > 1:
            probs_x_ulb = concat_all_gather(probs_x_ulb)
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1,keepdim=True)

        if algorithm.use_quantile:
            self.time_p = self.time_p * self.m + (1 - self.m) * torch.quantile(max_probs,0.8) #* max_probs.mean()
        else:
            self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean()
        
        if algorithm.clip_thresh:
            self.time_p = torch.clip(self.time_p, 0.0, 0.95)

        self.p_model = self.p_model * self.m + (1 - self.m) * probs_x_ulb.mean(dim=0)
        hist = torch.bincount(max_idx.reshape(-1), minlength=self.p_model.shape[0]).to(self.p_model.dtype) 
        self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())

        algorithm.p_model = self.p_model 
        algorithm.label_hist = self.label_hist 
        algorithm.time_p = self.time_p 
    

    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(logits_x_ulb.device)
        if not self.p_model_2.is_cuda:
            self.p_model_2 = self.p_model_2.to(logits_x_ulb.device)
        if not self.label_hist.is_cuda:
            self.label_hist = self.label_hist.to(logits_x_ulb.device)
        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        self.update(algorithm, probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
        # mask = max_probs.ge(0.99).to(max_probs.dtype)
        # self.pos_th=self.time_p*mod
        # return mask
        pos_tmp_the=(1-self.neg_th.sum(dim=1))
        #total_ratio=1-self.tau
        #total_ratio=total_ratio/total_ratio.max()
        if algorithm.args.use_mod:
            self.pos_th=pos_tmp_the*mod
        else:
            self.pos_th=pos_tmp_the
        #self.pos_th=(1-self.tau)
        #self.p_model_2 = self.p_model_2 * 0.9994 + (1 - 0.9994) * self.pos_th
        mask = max_probs.ge(self.pos_th[max_idx]).to(max_probs.dtype)
        # mask = max_probs.ge(0.99).to(max_probs.dtype)
        return mask

    @torch.no_grad()
    def get_th(self,algorithm):
        return self.pos_th,self.neg_th
    
    @torch.no_grad()
    def get_positive_mask(self,algorithm,logits_x_ulb):
        probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        max_probs, max_idx = probs_x_ulb.max(dim=-1)

        pos_tmp_the=(1-self.neg_th.sum(dim=1))

        mask = max_probs.ge(pos_tmp_the[max_idx]).to(max_probs.dtype)
        return mask
    


        
