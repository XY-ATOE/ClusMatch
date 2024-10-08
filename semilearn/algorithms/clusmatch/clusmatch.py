
import torch
import torch.nn.functional as F

from .utils import ClusMatchThresholingHook
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
import numpy as np




# TODO: move these to .utils or algorithms.utils.loss
def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val

def entropy_loss(mask, logits_s, prob_model, label_hist):
    mask = mask.bool()

    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model 
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()

def getnegativelog(logits_x_ulb_s):
    maxx, _ = torch.max(logits_x_ulb_s, dim=-1)
    new_exps=(logits_x_ulb_s-maxx.unsqueeze(dim=1).expand_as(logits_x_ulb_s)).exp()
    sumexps=new_exps.sum(dim=1).unsqueeze(dim=1).expand_as(logits_x_ulb_s)
    return (sumexps-new_exps).log()-sumexps.log()

def negative_loss(positive_mask,negative_mask,logits_x_ulb_s):
    probs = torch.softmax(logits_x_ulb_s, dim=-1)
    
    acceptmask=(negative_mask).sum(dim=1).ge(0.5).to(logits_x_ulb_s.dtype)
    all_negative_loss = -(1 - (negative_mask * probs).sum(1)).clamp(1e-8,1).log()
    negative_loss = all_negative_loss*acceptmask#*(1/neg_mask_cnt)


    loss = negative_loss.mean()#+h_ne#+2.0*positive_loss.mean()
    if torch.isnan(negative_loss).sum() > 0:
        negative_loss=replace_inf_to_zero(negative_loss)
        print(negative_loss)

    return loss
def pairwise_cosine(x1: torch.Tensor, x2: torch.Tensor, pairwise=True):
    x1 = F.normalize(x1)
    x2 = F.normalize(x2)
    if not pairwise:
        return (1 - (x1 * x2).sum(dim=1))
    return 1 - x1.mm(x2.T)


@ALGORITHMS.register('clusmatch')
class ClusMatch(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(T=args.T, hard_label=args.hard_label, ema_p=args.ema_p, use_quantile=args.use_quantile, clip_thresh=args.clip_thresh)
        self.lambda_e = args.ent_loss_ratio
        self.lambda_n = args.neg_loss_ratio
        self.cnt=0

    def init(self, T, hard_label=True, ema_p=0.999, use_quantile=True, clip_thresh=False):
        self.T = T
        self.use_hard_label = hard_label
        self.ema_p = ema_p
        self.use_quantile = use_quantile
        self.clip_thresh = clip_thresh


    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        print('ema',self.args.ema_p)
        self.register_hook(ClusMatchThresholingHook(num_classes=self.num_classes, momentum=self.args.ema_p), "MaskingHook")
        super().set_hooks()


    def train_step(self,idx_lb, x_lb, y_lb, idx_ulb,x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}
            
            
            sup_loss = self.ce_loss(logits_x_lb, y_lb.long(), reduction='mean')
            
            neg_mask = self.call_hook("getnegmask", "MaskingHook", logits_x_ulb=logits_x_ulb_w)
            
            # calculate mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w)
            
            
            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)
            
            cluster_pseudo_label = self.cluster_label_for_train[idx_ulb]
            
            posth,negth=self.call_hook("get_th", "MaskingHook")
            
            pseudo_label_onehot=torch.zeros_like(logits_x_ulb_s).cuda().scatter_(1,pseudo_label.view(pseudo_label.shape[0],1),1)
            cluster_pseudo_label_onehot=torch.zeros_like(logits_x_ulb_s).cuda().scatter_(1,cluster_pseudo_label.view(pseudo_label.shape[0],1),1)
            mix_pseudo_label_onehot = self.lambda_u*pseudo_label_onehot + self.cluster_pseudo_label_weight*cluster_pseudo_label_onehot
            
            
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                          mix_pseudo_label_onehot,
                                          'ce',
                                          mask=mask)
            
            
            equal_num_ratio=(pseudo_label==cluster_pseudo_label).sum()/pseudo_label.shape[0]
            neg_mask=neg_mask.scatter(1,cluster_pseudo_label.view(cluster_pseudo_label.shape[0],1),0.0).to(neg_mask.dtype)
            neg_loss=negative_loss(mask,neg_mask,logits_x_ulb_s)

            
            # calculate entropy loss
            if mask.sum() > 0:
               ent_loss, _ = entropy_loss(mask, logits_x_ulb_s, self.p_model, self.label_hist)
            else:
               ent_loss = 0.0
            #ent_loss = 0.0
            

            total_loss = sup_loss + unsup_loss + self.lambda_e * ent_loss + self.lambda_n*neg_loss# + (posth-negth).mean()# + self.lambda_c*contrastive_loss
 
        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item(),
                                         neg_loss=neg_loss.item(),
                                         neg_mask=neg_mask.float().sum(dim=1).mean().item(),
                                         #contrastive_loss=contrastive_loss.item(),
                                         equal_num_ratio=equal_num_ratio.item(),
                                         cluster_pseudo_label_weight=self.cluster_pseudo_label_weight,
                                         lambda_u=self.lambda_u,
                                         posth_mean=posth.mean().item(),
                                         negth_mean=negth.mean().item(),
                                         posth_std=posth.std().item(),
                                         negth_std=negth.std().item(),
                                         posth_max=posth.max().item(),
                                         negth_max=negth.max().item(),
                                         posth_min=posth.min().item(),
                                         negth_min=negth.min().item(),
                                         neg_mask_max=self.cnt
                                         #lambda_c=self.lambda_c
                                         )
        
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['MaskingHook'].p_model.cpu()
        save_dict['time_p'] = self.hooks_dict['MaskingHook'].time_p.cpu()
        save_dict['label_hist'] = self.hooks_dict['MaskingHook'].label_hist.cpu()
        save_dict['all_tau'] = self.hooks_dict['MaskingHook'].all_tau.cpu()
        save_dict['tau'] = self.hooks_dict['MaskingHook'].tau.cpu()
        #save_dict['topk'] = self.hooks_dict['MaskingHook'].topk.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].time_p = checkpoint['time_p'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].label_hist = checkpoint['label_hist'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].all_tau = checkpoint['all_tau'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].tau = checkpoint['tau'].cuda(self.args.gpu)
        #self.hooks_dict['MaskingHook'].topk = checkpoint['topk'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--ent_loss_ratio', float, 0.01),
            SSL_Argument('--use_quantile', str2bool, False),
            SSL_Argument('--clip_thresh', str2bool, False),
        ]