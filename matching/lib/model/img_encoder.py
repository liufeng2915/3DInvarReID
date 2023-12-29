import torchvision
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class GeMPooling(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), x.size()[2:]).pow(1./self.p)


class MaxAvgPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpooling = nn.AdaptiveMaxPool2d(1)
        self.avgpooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        max_f = self.maxpooling(x)
        avg_f = self.avgpooling(x)

        return torch.cat((max_f, avg_f), 1)
        

class ImageEncoder(nn.Module):
    def __init__(self, feat_dim=4096, dim_naked_shape=512, dim_clohted_shape=512, dim_texture=512):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        resnet50.layer4[0].conv2.stride=(1, 1)
        resnet50.layer4[0].downsample[0].stride=(1, 1) 
        
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.globalpooling = MaxAvgPooling()
        self.bn = nn.BatchNorm1d(feat_dim)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

        # naked (id) shape code
        self.id_fc = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(),
            nn.Linear(512, dim_naked_shape),
        )

        self.cloth_module = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True), nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=True),

        )
        # clothed shape code
        self.cloth_fc = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, dim_clohted_shape),
        )

        # texture code
        self.tex_fc = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, dim_texture),
        )

    def forward(self, infer_matching, x):
        x = self.base(x)
        x_ori = x
        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)
        f = self.bn(x)

        #
        f_cloth = self.cloth_module(x_ori)
        f_cloth = F.avg_pool2d(f_cloth, f_cloth.size()[2:])
        f_cloth = f_cloth.view(f_cloth.size(0), -1)

        #
        if infer_matching:
            return f
        else:
            z_naked_shape = self.id_fc(x)
            z_clothed_shape = self.cloth_fc(f_cloth)
            z_texture = self.tex_fc(f_cloth)
            return f, z_naked_shape, z_clothed_shape, z_texture