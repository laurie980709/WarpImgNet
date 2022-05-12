import argparse
class Options:
    def __init__(self):
        self.opt = None
        self.parse()
    def parse(self):
        parser = argparse.ArgumentParser(description="Argparse of dataset")
        parser.add_argument('--data_root',type=str, default='/home/yl/YL/lsc-cnn-master-pre/dataset/ST_partA',help='root of dataset')
        parser.add_argument('--is_train',type=int,default=2,help='0:test,1: eval,2: train')
        parser.add_argument('--resize', type=str, default='512 512',help='size of resized img ')
        parser.add_argument('--train_epoch',type=int,default=200,help='how many epoch you want to train(only is_train=1,valid)')
        parser.add_argument('--batch_size',type=int,default=1,help='batch size of models')
        self.opt = parser.parse_args()