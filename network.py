import torch.nn as nn
import torch
import torchvision.models as models
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # vgg16 = models.vgg16(pretrained=True).features
        # conv_modules = [m for m in vgg16]
        # self.vgg_conv = nn.Sequential(*conv_modules[:10])
        # for p in self.vgg_conv.parameters():
        #     p.requires_grad = False

        self.cov1_down = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64,track_running_stats=True),
            # nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64, track_running_stats=True),
            # nn.MaxPool2d(2)
        )
        self.cov2_down = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32,track_running_stats=True),
            # nn.Conv2d(32, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32, track_running_stats=True),
            # nn.MaxPool2d(2)
        )
        self.cov3_down = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16,track_running_stats=True),
            # nn.Conv2d(16, 16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16, track_running_stats=True),
            # nn.MaxPool2d(2)
        )
        self.cov4_down = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            # nn.BatchNorm2d(8, track_running_stats=True),
            # nn.Conv2d(8, 8, kernel_size=3, padding=1),
            # nn.BatchNorm2d(8, track_running_stats=True),
            # nn.MaxPool2d(2)
        )
        self.cov5_down = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            # nn.BatchNorm2d(4,track_running_stats=True),
            # nn.Conv2d(4, 4, kernel_size=3, padding=1),
            # nn.BatchNorm2d(4, track_running_stats=True),
            # nn.MaxPool2d(2)
        )
        self.cov6_down = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=3, padding=1),
            # nn.BatchNorm2d(2,track_running_stats=True),
            # nn.Conv2d(2, 2, kernel_size=3, padding=1),
            # nn.BatchNorm2d(2, track_running_stats=True),
            # nn.MaxPool2d(2)
        )
        self.cov7_down = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
        )
        self.aap = nn.Sequential(
            nn.AdaptiveAvgPool2d((128, 1)),
            # nn.BatchNorm2d(1, track_running_stats=True)
        )
        self.fc = nn.Linear(128, 1)#wo hao ai ni

    def forward(self,input):
        batch = input.shape[0]
        # feat_map = self.vgg_conv(input) #[1,128,h/2,w/2]
        feat_map = input
        x = self.cov1_down(feat_map)
        x = self.cov2_down(x)
        x = self.cov3_down(x)
        x = self.cov4_down(x)
        x = self.cov5_down(x)
        x = self.cov6_down(x)
        x = self.cov7_down(x) #[1,1,h/2,w/2]
        # print(x.shape)
        x = self.aap(x)
        # print(x.shape)
        # x = torch.squeeze(x,dim = -1)
        x = x.view(batch, -1)
        # x = x.transpose(1, 2)
        # x = torch.squeeze(x, dim=-1)
        # x = torch.squeeze(x, dim=-1)
        x = self.fc(x)
        # x = torch.unsqueeze(x, dim = -1)
        return {'scale_vec':x, 'feat_map':feat_map}

if __name__=='__main__':
    input = np.random.rand(1, 3, 512, 512)
    input = torch.Tensor(input)
    net = Network()
    res = net(input)
