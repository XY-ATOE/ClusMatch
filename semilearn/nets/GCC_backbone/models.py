"""
Author: Huasong Zhong
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_cifar import resnet18 as resnet18cifar
from .resnet_stl import resnet18 as resnet18stl
from .resnet_stl import resnet34 as resnet34stl
from .resnet_propos import ResNet as resnetpropos
from .resnet import resnet18 as oriresnet18
from .resnet import resnet50 as oriresnet50


class SeCuImagenetModel(nn.Module):
    def __init__(self, num_classes,backbone):
        super(SeCuImagenetModel, self).__init__()
        # self.t = 0.05
        self.encoder = backbone['backbone']
        self.backbone_dim = 128
        dim_mlp = 2048
        dim = 128
        self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp),
                                        nn.ReLU(inplace=True), nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp),
                                        nn.ReLU(inplace=True), nn.Linear(dim_mlp, dim), nn.BatchNorm1d(dim))
        # prediction head
        self.predictor = nn.Sequential(nn.Linear(dim, dim_mlp), nn.BatchNorm1d(dim_mlp),
                                       nn.ReLU(inplace=True), nn.Linear(dim_mlp, dim))
        # self.center_0 = nn.Linear(self.backbone_dim, num_classes)
        # self.center_0 = nn.Parameter(torch.randn(self.backbone_dim, num_classes))
        self.cluster_head_new =nn.Linear(dim, num_classes)
        self.register_parameter("center_" + str(0), nn.Parameter(F.normalize(torch.randn(self.backbone_dim , num_classes), dim=0)))
        nn.init.xavier_normal_(self.cluster_head_new.weight.data)
        self.cluster_head_new.bias.data.zero_()
        
    def forward(self, x,change=False):
        #features = self.contrastive_head(self.backbone(x))
        features = self.encoder(x)
        # features = self.predictor(features)
        # features = self.encoder.fc(middle)
        features = F.normalize(features, dim=1)
        cluster_outs = self.cluster_head_new(features)
        
        # cluster_outs = features @ F.normalize(self.center_0, dim=0) / 0.05
        result_dict = {'feat':features, 'logits':cluster_outs}
        return result_dict


class SeCuEnd2EndModel(nn.Module):
    def __init__(self, num_classes,backbone):
        super(SeCuEnd2EndModel, self).__init__()
        
        self.encoder = backbone['backbone']
        self.backbone_dim = 128
        dim_mlp = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp),
                                        nn.ReLU(inplace=True), nn.Linear(dim_mlp, self.backbone_dim))
        # self.center_0 = nn.Linear(self.backbone_dim, num_classes)
        # self.center_0 = nn.Parameter(torch.randn(self.backbone_dim, num_classes))
        self.cluster_head_new =nn.Linear(self.backbone_dim, num_classes)
        self.register_parameter("center_" + str(0), nn.Parameter(F.normalize(torch.randn(self.backbone_dim , num_classes), dim=0)))
        # nn.init.xavier_normal_(self.cluster_head_new.weight.data)
        # self.cluster_head_new.bias.data.zero_()

    def forward(self, x,change=False):
        #features = self.contrastive_head(self.backbone(x))
        middle = self.encoder(x)
        features = self.encoder.fc(middle)
        features = F.normalize(features, dim=1)
        cluster_outs = self.cluster_head_new(features)
        
        # cluster_outs = features @ F.normalize(self.center_0, dim=0) / 0.05
        result_dict = {'feat':features, 'logits':cluster_outs}
        return result_dict



class End2EndModel(nn.Module):
    def __init__(self, num_classes,backbone):
        super(End2EndModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.cluster_head_new =nn.Linear(self.backbone_dim, num_classes)
        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, num_classes) for _ in range(1)])
        # nn.init.xavier_normal_(self.cluster_head_new.weight.data)
        # self.cluster_head_new.bias.data.zero_()

    def forward(self, x,change=False):
        #features = self.contrastive_head(self.backbone(x))
        features = self.backbone(x)
        if change==False:
            cluster_outs = self.cluster_head_new(features)
        else:
            cluster_outs = [cluster_head(features) for cluster_head in self.cluster_head][0]
        result_dict = {'feat':features, 'logits':cluster_outs}
        return result_dict

class End2EndModel2(nn.Module):
    def __init__(self, num_classes,backbone):
        super(End2EndModel2, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.contrastive_head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.ReLU(), nn.Linear(self.backbone_dim, 128))
        self.cluster_head_new = nn.Linear(128, num_classes)
        #self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, num_classes) for _ in range(1)])
        nn.init.xavier_normal_(self.cluster_head_new.weight.data)
        self.cluster_head_new.bias.data.zero_()

    def forward(self, x,change=False):
        #features = self.contrastive_head(self.backbone(x))
        features = self.backbone(x)
        middle_features = self.contrastive_head(features)
        cluster_outs = self.cluster_head_new(middle_features)
        result_dict = {'feat':features, 'logits':cluster_outs}
        return result_dict

class End2EndModelForPropos2(nn.Module):
    def __init__(self, num_classes,backbone,backbone_dim=512):
        super(End2EndModelForPropos2, self).__init__()
        self.encoder_k = backbone
        self.backbone_dim = backbone_dim
        hidden_size=4096
        fea_dim=256
        self.projector_k = nn.Sequential(
            nn.Linear(self.backbone_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, fea_dim)
        )
        self.fea_dim=fea_dim
        self.num_classes=num_classes
        self.cluster_head = nn.Linear(fea_dim, num_classes)
        
        nn.init.xavier_normal_(self.cluster_head.weight.data)
        self.cluster_head.bias.data.zero_()

        # self.cluster_head_new = nn.Linear(fea_dim, num_classes)
        # nn.init.xavier_normal_(self.cluster_head_new.weight.data)
        # self.cluster_head_new.bias.data.zero_()
        
    def init_cluster_head(self):
        self.cluster_head = nn.Linear(self.fea_dim, self.num_classes)
        nn.init.xavier_normal_(self.cluster_head.weight.data)
        self.cluster_head.bias.data.zero_()
    def forward(self, x,change=False):
        #features = self.contrastive_head(self.backbone(x))
        features = self.encoder_k(x)
        
        middle_features = self.projector_k(features)
        #print(middle_features.shape)
        cluster_outs = self.cluster_head(middle_features)
        result_dict = {'feat':middle_features, 'logits':cluster_outs}
        return result_dict

class End2EndModelForPropos1(nn.Module):
    def __init__(self, num_classes,backbone):
        super(End2EndModelForPropos1, self).__init__()
        self.encoder_k = backbone
        self.backbone_dim = 512
        hidden_size=4096
        fea_dim=256
        
        self.cluster_head = nn.Linear(self.backbone_dim, num_classes)
        nn.init.xavier_normal_(self.cluster_head.weight.data)
        self.cluster_head.bias.data.zero_()
    def forward(self, x,change=False):
        #features = self.contrastive_head(self.backbone(x))
        features = self.encoder_k(x)
        
        cluster_outs = self.cluster_head(features)
        result_dict = {'feat':features, 'logits':cluster_outs}
        return result_dict


def SeCu_resnet18_cifar(pretrained=False, pretrained_path=None, **kwargs):
    model = SeCuEnd2EndModel(backbone=resnet18cifar(),**kwargs)
    return model
def SeCu_resnet50_imagent(pretrained=False, pretrained_path=None, **kwargs):
    model = SeCuImagenetModel(backbone=oriresnet50(),**kwargs)
    return model

def GCC_resnet18_cifar(pretrained=False, pretrained_path=None, **kwargs):
    model = End2EndModel(backbone=resnet18cifar(),**kwargs)
    return model

def GCC_resnet18_stl(pretrained=False, pretrained_path=None, **kwargs):
    model = End2EndModel(backbone=resnet18stl(feature_size=5),**kwargs)
    #print(model)
    return model

def propos_resnet18_cifar2(pretrained=False, pretrained_path=None, **kwargs):
    model = End2EndModelForPropos2(backbone=resnetpropos('resnet18',cifar=True)(),**kwargs)
    return model

def propos_resnet18_cifar1(pretrained=False, pretrained_path=None, **kwargs):
    model = End2EndModelForPropos1(backbone=resnetpropos('resnet18',cifar=True)(),**kwargs)
    return model

def propos_resnet34_cifar(pretrained=False, pretrained_path=None, **kwargs):
    model = End2EndModelForPropos2(backbone=resnetpropos('resnet34',cifar=True)(),**kwargs)
    return model
def propos_resnet50_imagent(pretrained=False, pretrained_path=None, **kwargs):
    model = End2EndModelForPropos2(backbone=resnetpropos('resnet50')(),backbone_dim=2048,**kwargs)
    return model


def propos_resnet18_stl(pretrained=False, pretrained_path=None, **kwargs):
    model = End2EndModelForPropos2(backbone=resnet18stl()['backbone'],**kwargs)
    return model

def propos_resnet34_stl(pretrained=False, pretrained_path=None, **kwargs):
    model = End2EndModelForPropos2(backbone=resnet34stl()['backbone'],**kwargs)
    return model


def propos_resnet18_stl_for_size64(pretrained=False, pretrained_path=None, **kwargs):
    model = End2EndModelForPropos2(backbone=resnet18stl(feature_size=5)['backbone'],**kwargs)
    return model

def propos_oriresnet18(pretrained=False, pretrained_path=None, **kwargs):
    model = End2EndModelForPropos2(backbone=resnetpropos('resnet18')(),**kwargs)
    return model


# def DINO_vit_b_16(pretrained=False, pretrained_path=None, **kwargs):
#     model = End2EndModel(backbone=resnetpropos('resnet18')(),**kwargs)
#     return model