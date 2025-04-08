import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import argparse
import dataset
from torch.utils.data import DataLoader
from get_model import get_model
import datetime
import time

class Test(object):
    def __init__(self, Dataset, Path, model, checkpoint, task):
        ## task
        self.task = task 

        ## dataset
        self.cfg    = Dataset.Config(datapath=Path, snapshot=checkpoint, mode='test')
        self.data   = Dataset.Data(self.cfg, model)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)

        ## network
        self.net = get_model(self.cfg, model)
        self.net.train(False)
        self.net.cuda()

    def save(self):
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image, shape  = image.cuda().float(), (H, W)
                out = self.net(image, shape, name)
                pred = torch.sigmoid(out[0,0]).cpu().numpy()*255
                head = 'Prediction/'+model+'/'+ self.cfg.datapath.split('/')[-2]
                if not os.path.exists(head):
                    print("create a new folder: {}".format(head))
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='TRSNet')
    parser.add_argument("--task", default='SOD')
    parser.add_argument("--ckpt", default='checkpoint/TRSNet/TRSNet')
    
    args   = parser.parse_args()
    task   = args.task
    model  = args.model
    ckpt   = args.ckpt
    
    print(args.model, args.ckpt)
    for path in ['datasets/ECSSD/Test', 'datasets/PASCAL-S/Test', 'datasets/DUTS/Test', 'datasets/HKU-IS/Test', 'datasets/DUT-OMRON/Test', 'datasets/SOD/Test']:
        t = Test(dataset, path, model, ckpt, task)
        t.save()

