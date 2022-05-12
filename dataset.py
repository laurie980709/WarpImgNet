import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import argparse
from torchvision import transforms
from PIL import Image
from option import *


class Our_Dataset(Dataset):
    def __init__(self,opt):
        #self.opt = opt
        self.hw =[int(i) for i in opt.resize.split(' ')]
        self.data_root = opt.data_root
        self.is_train = opt.is_train
        if self.is_train==2:
            self.img_dir = os.path.join(self.data_root,'train_data','images')
        elif self.is_train==0:
            self.img_dir = os.path.join(self.data_root, 'test_data', 'images')
        elif self.is_train==1:
            self.img_dir = os.path.join(self.data_root, 'val_data', 'images')
        img_list_local = os.listdir(self.img_dir)
        self.img_list = [os.path.join(self.img_dir,img) for img in img_list_local]
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transformer(img,self.hw)
        return {'img':img}
    def __len__(self):
        return len(self.img_list)
    def transformer(self,img,hw):
        transform = transforms.Compose([
            transforms.Resize((hw[0],hw[1])),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5,0.43,0.44],std=[0.2,0.2,0.2])
        ])
        img = transform(img)
        # img = vgg_conv(img)
        return img


if __name__ == '__main__':
    sparse = Options()
    opt = sparse.opt
    ds = Our_Dataset(opt)
    itm=ds[10]
    print(itm['img'].shape)