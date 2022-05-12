import torch
from option import Options
from dataset import Our_Dataset
from torch.utils.data import DataLoader
from network import Network
from loss import VarLoss
from tqdm import tqdm
import time
import pickle
import os
import torch.nn as nn
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment='test_tensorboard')

# Press the green button in the gutter to run the script.
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    vgg16 = models.vgg16(pretrained=True).features
    conv_modules = [m for m in vgg16]
    vgg_conv = nn.Sequential(*conv_modules[:10]).cuda()
    for p in vgg_conv.parameters():
        p.requires_grad = False
    sparse = Options()
    opt = sparse.opt
    if(opt.is_train==2):

        opt.is_train = 1
        ds_val = Our_Dataset(opt)
        data_loader_val = DataLoader(dataset=ds_val,
                                 batch_size=opt.batch_size,
                                 shuffle=False)
        opt.is_train = 2
        ds_train = Our_Dataset(opt)
        data_loader_train = DataLoader(dataset=ds_train,
                                 batch_size=opt.batch_size,
                                 shuffle=True)
    if(opt.is_train == 0):
        ds_test = Our_Dataset(opt)
        data_loader_test = DataLoader(dataset=ds_test,
                                 batch_size=opt.batch_size,
                                 shuffle=False)
    model = Network().to(device)
    criterion = VarLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for param in model.parameters():
        param.requires_grad = (opt.is_train==2)
    if (opt.is_train==2):
        #train
        for epoch in range(opt.train_epoch):
            for phase in ['train','val']:
                if phase == 'train':
                    loader = data_loader_train
                else:
                    loader = data_loader_val
                print("-------------epoch:", epoch, phase," start--------------")
                start = time.time()
                running_loss = 0.0
                pbar = tqdm(enumerate(loader), total=len(loader))
                for _, mini_batch in pbar:
                    img = mini_batch['img'].to(device)
                    img = vgg_conv(img)
                    optimizer.zero_grad()
                    res = model(img)
                    loss_itm = criterion(res['scale_vec'], res['feat_map'])
                    running_loss += loss_itm.item()
                    if phase =='train':
                        loss_itm.backward()
                        optimizer.step()
                end = time.time()
                print('\n'+phase + "--loss:", running_loss / len(loader), "time spend:%.2fs" % (end - start))
                if phase == 'train':
                    writer.add_scalar('train',running_loss / len(loader),epoch)
                else:
                    writer.add_scalar('val', running_loss / len(loader),epoch)
            if epoch%20==0:
                torch.save(model,"./checkpoints/checkpoint-%d.pth"%epoch)
                print('save my.pth successfully!')
    else:
        with torch.no_grad():
            model = torch.load('./checkpoints/ST_B_new/checkpoint-160.pth')
            model.eval()
            model.to(device)
            start = time.time()
            running_loss = 0.0
            loader = data_loader_test
            pbar = tqdm(enumerate(loader), total=len(loader))
            scale = []
            for _, mini_batch in pbar:
                img = mini_batch['img'].to(device)
                # optimizer.zero_grad()
                img = vgg_conv(img)
                res = model(img)
                scale.append(res['scale_vec'].cpu().numpy())
                # print(res['scale_vec'])
                loss_itm = criterion(res['scale_vec'], res['feat_map'])
                running_loss += loss_itm.item()
            end = time.time()
            with open('./scale_map.save', 'wb') as fp:
                pickle.dump(scale, fp, protocol=pickle.HIGHEST_PROTOCOL)
            print("\ntest--loss:", running_loss / len(loader), "time spend:%.2fs" % (end - start))
if __name__ == '__main__':
    main()
