import torch

import numpy as np

class VarLoss(torch.nn.Module):
    def __init__(self):
        super(VarLoss,self).__init__()
    def forward(self,scale,feat_map):
        D = torch.sum(torch.sum(feat_map, dim=-1, keepdim=True), dim=-3, keepdim=True) / feat_map.shape[3]#1,1,128,1
        D = D.squeeze(-1)
        # print(D.shape)
        batch = scale.shape[0]
        h = feat_map.shape[2]
        scale_vec = torch.zeros(batch, 1, h).cuda()
        for b in range(batch):
            scale_vec[b] = torch.range(feat_map.shape[2], 1, -1)
            # print(scale_vec[b])
            # a = torch.log(scale_vec[b]+1e-10)
            # c = torch.log(1+scale[b])
            # scale_vec[b] = torch.log(scale_vec[b]+scale[b])/torch.log(1+scale[b])
            scale_vec[b] = torch.log(scale_vec[b] + scale[b])
            # scale_vec[b] = torch.log(scale_vec[b]+10000000)

        # print(scale_vec)
        # print(scale_vec.shape)
        D = D / scale_vec
        loss_itm = torch.var(D, dim=2)
        loss_itm = torch.sum(loss_itm, dim=0)
        return loss_itm


if __name__ == '__main__':
    feat = torch.rand(2, 1, 128, 128)
    scale = torch.rand(2, 1)
    vl = VarLoss()
    res = vl(scale, feat)
