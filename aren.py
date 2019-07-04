#coding:utf-8
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import init
import torch
import datetime
from torch.autograd import Variable
class AREN(nn.Module):
    def __init__(self, cfg, model, train_attr, test_attr):
        super(AREN, self).__init__()

        c1,d = train_attr.size()
        c2,_ = test_attr.size()
        self.train_linear = nn.Linear(d,c1,False)
        self.train_linear.weight = train_attr
        for para in self.train_linear.parameters():
            para.requires_grad = False

        self.test_linear = nn.Linear(d,c2,False)
        self.test_linear.weight = test_attr
        for para in self.test_linear.parameters():
            para.requires_grad = False

        # Resnet101
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.classifier = nn.Sequential(*list(model.children())[-2:-1])
        self.pre_layer = nn.ModuleList([self.train_linear,self.test_linear,
                                        self.features,self.classifier])

        pre_nodes = 2048
        self.cov_channel = 2048

        if cfg['fix_feature'] > 0:
            print('base layer network weight has freezing ....')
            for param in self.features.parameters():
                param.requires_grad = False
        print('Extract the image features : {}'.format(pre_nodes))

        drop_out = cfg['dropout']

        # p stream
        self.map_threshold = cfg['threshold']
        self.parts = cfg['parts']
        self.map_size = 7
        self.pool = nn.MaxPool2d(self.map_size,self.map_size)
        self.cov = nn.Conv2d(self.cov_channel,self.parts,1)
        self.p_linear = nn.Linear(self.cov_channel*self.parts,d,False)
        self.dropout2 = nn.Dropout(drop_out)

        # b stream
        self.compress_map = cfg['cpp_map']
        self.conv_bilinear = nn.Conv2d(self.cov_channel,self.compress_map,1)
        self.b_linear = nn.Linear(self.compress_map*self.cov_channel,d,False)
        self.dropout3 = nn.Dropout(drop_out)
        self.coef = self.map_size*self.map_size

    def forward(self, x):
        features = self.pre_layer[2](x)
        w = features.size()
        weights = torch.sigmoid(self.cov(features)) # batch x parts x 7 x 7

        # threshold the weights
        batch,parts,width,height = weights.size()
        weights_layout = weights.view(batch,-1)
        threshold_value,_ = weights_layout.max(dim=1)
        local_max,_ = weights.view(batch,parts,-1).max(dim=2)
        threshold_value = self.map_threshold*threshold_value.view(batch,1) \
            .expand(batch,parts)
        weights = weights*local_max.ge(threshold_value).view(batch,parts,1,1). \
            float().expand(batch,parts,width,height)

        blocks = []
        blocks_Z = []
        X = self.conv_bilinear(features).view(w[0], self.compress_map, self.coef)
        for k in range(self.parts):
            Y = features*weights[:,k,:,:]. \
                unsqueeze(dim=1). \
                expand(w[0],self.cov_channel,w[2],w[3])
            blocks.append(self.pool(Y).squeeze().view(-1,self.cov_channel))
            Y = Y.view(w[0],w[1],7**2)
            Z = torch.bmm(X, torch.transpose(Y, 1, 2)) / self.coef  # Bilinear
            blocks_Z.append(Z.view(w[0], -1))

        p_output = self.dropout2(self.p_linear(torch.cat(blocks, dim=1)))
        b_output = self.dropout3(self.b_linear(torch.max(torch.stack(blocks_Z), 0)[0]))

        if self.training:
            p_output = self.pre_layer[0](p_output)
            b_output = self.pre_layer[0](b_output)
        else:
            p_output = self.pre_layer[1](p_output)
            b_output = self.pre_layer[1](b_output)

        return p_output, b_output



