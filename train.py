import os
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import dataset  
import argparse
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from get_model import get_model


# IoU Loss
def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()


def train(Dataset, parser):
    
    args   = parser.parse_args()
    _MODEL_ = args.model
    _DATASET_ = args.dataset
    _LR_ = args.lr
    _DECAY_ = args.decay
    _MOMEN_ = args.momen
    _BATCHSIZE_ = args.batchsize
    _EPOCH_ = args.epoch
    _LOSS_ = args.loss
    _SAVEPATH_ = args.savepath
    _VALID_ = args.valid 
    print('Args: ',args)



    cfg    = Dataset.Config(datapath=_DATASET_, savepath=_SAVEPATH_, mode='train', batch=_BATCHSIZE_, lr=_LR_, momen=_MOMEN_, decay=_DECAY_, epoch=_EPOCH_)
    data   = Dataset.Data(cfg, _MODEL_)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True, num_workers=6)

    ## network
    net = get_model(cfg, _MODEL_)
    net.train(True)
    net.cuda()

    ## parameter
    base, head = [], []
    
    for name, param in net.named_parameters():
        if 'encoder.conv1' in name or 'encoder.bn1' in name:
            pass
        elif 'encoder' in name:
            base.append(param)
        elif 'network' in name:
            base.append(param)     
        else:
            head.append(param)

    optimizer = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    scaler = GradScaler()
    sw             = SummaryWriter(cfg.savepath)
    global_step    = 0
 
    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step, (image, mask) in enumerate(loader):
            image, mask = image.cuda(), mask.cuda() 
            # with autocast():
            with autocast():
                image = image.to(dtype=torch.float32)         
                out = net(image)          
                loss  = F.binary_cross_entropy_with_logits(out, mask) + iou_loss(out, mask)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()  
            scaler.step(optimizer)  
            scaler.update()  
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            ## log
            global_step += 1
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss':loss.item()}, global_step=global_step)

            if step%10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f'
                    %(datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item()))

        if epoch > -1:
            torch.save(net.state_dict(), cfg.savepath+'/'+_MODEL_+str(epoch+1))
            for path in ['datasets/ECSSD/Test', 'datasets/PASCAL-S/Test']:
                t = Valid(dataset, path, epoch, _MODEL_, _SAVEPATH_, "SOD")
                t.save()
            cmd = os.path.join(os.getcwd().split('main')[0], "run_sod_valid.sh")
            os.system('{} {}'.format('sh', cmd+' TRSNet-valid-'+str(epoch+1)))                
                       

class Valid(object):
    def __init__(self, Dataset, Path, epoch, model_name, checkpoint_path, task=None):
        ## dataset
        self.cfg    = Dataset.Config(datapath=Path, snapshot=checkpoint_path+model_name+str(epoch+1), mode='test')
        self.data   = Dataset.Data(self.cfg, model_name)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)

        ## network
        self.net = get_model(self.cfg, model_name)
        self.net.train(False)
        self.net.cuda()
        self.epoch = epoch

        ## task
        self.task = task


    def save(self):
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image, shape  = image.cuda().float(), (H, W)
                out = self.net(image, shape)
                pred = torch.sigmoid(out[0,0]).cpu().numpy()*255
                head = 'Prediction/TRSNet-valid-'+str(self.epoch+1)+'/'+ self.cfg.datapath.split('/')[-2]

                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))
 

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='TRSNet')
    parser.add_argument("--dataset", default='../datasets/DUTS/Train')
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momen", type=float, default=0.9)  
    parser.add_argument("--decay", type=float, default=1e-4)  
    parser.add_argument("--batchsize", type=int, default=14)  
    parser.add_argument("--epoch", type=int, default=60)  
    parser.add_argument("--loss", default='CPR')  
    parser.add_argument("--savepath", default='../checkpoint/TRSNet')  
    parser.add_argument("--valid", default=True)  
    train(dataset, parser)
