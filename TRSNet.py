import torch
from torch import nn
from pvtv2_encoder import pvt_v2_b5
from modules import CFRB, CCG, ASCA, FAMHA
from timm.models import create_model
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()


class TRSNet(torch.nn.Module):
    def __init__(self, cfg, model_name='TRSNet'):
        super(TRSNet, self).__init__()
        self.cfg = cfg
        self.model_name = model_name        

        self.encoder = pvt_v2_b5()

        pretrained_dict = torch.load('checkpoint/Backbone/PVTv2/pvt_v2_b5.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(pretrained_dict)

        self.cfrb_conv1 = CFRB(feature_layer='1')
        self.cfrb_conv2 = CFRB(feature_layer='2')
        self.cfrb_conv3 = CFRB(feature_layer='3')
        self.cfrb_conv4 = CFRB(feature_layer='4')

        self.asca_1 = ASCA(inp = 64, oup = 64)
        self.asca_2 = ASCA(inp = 320, oup = 320)
        self.ccg = CCG(max_kernel=3)

        self.famha_f2 = FAMHA(d_model=320, d_k=320, d_v=320, h=8, H=48, W=48, ratio=2, apply_transform=True)
        self.famha_f3 = FAMHA(d_model=320, d_k=320, d_v=320, h=8, H=24, W=24, ratio=2, apply_transform=True)

        self.encoder_merge1_2 = nn.Sequential(nn.BatchNorm2d(384),
                                               nn.ConvTranspose2d(384, 192, kernel_size=3, padding=1, bias=True),
                                               nn.LeakyReLU())
        self.encoder_merge3_4 = nn.Sequential(nn.BatchNorm2d(832),
                                                nn.ConvTranspose2d(832, 416, kernel_size=3, padding=1, bias=True),
                                                nn.LeakyReLU())
        self.encoder_merge1234 = nn.Sequential(nn.BatchNorm2d(384),
                                                nn.ConvTranspose2d(384, 192, kernel_size=3, padding=1, bias=True),
                                                nn.LeakyReLU())
        
        self.encoder_mergeall = nn.Sequential(nn.BatchNorm2d(832),
                                        nn.ConvTranspose2d(832, 416, kernel_size=3, padding=1, bias=True),
                                        nn.LeakyReLU())


        self.trans_conv = nn.ConvTranspose2d(in_channels=416, out_channels=192, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(p=0.3)

        self.ff_conv_1 = nn.ConvTranspose2d(192, 1, kernel_size=3, padding=1)

        self.initialize()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x, shape=None, name=None):
        batch_size = x.size(0)

        features = self.encoder(x)

        x4 = features[0]
        x3 = features[1]
        x2 = features[2]
        x1 = features[3]

        conv1_cfrb_feats = self.cfrb_conv1(x1)
        conv2_cfrb_feats = self.cfrb_conv2(x2)
        conv3_cfrb_feats = self.cfrb_conv3(x3)
        conv4_cfrb_feats = self.cfrb_conv4(x4)

        f1 = self.asca_1(x1)
        f2 = self.asca_2(conv2_cfrb_feats)   
        f3 = self.ccg(conv3_cfrb_feats) 
        f4 = self.ccg(x4) 

        f2_reshaped = f2.permute(0, 2, 3, 1).reshape(batch_size, -1, 320)  
        f2_mhsa = self.famha_f2(f2_reshaped,f2_reshaped,f2_reshaped)
        f2_mhsa = f2_mhsa.reshape(batch_size, 48, 48, 320).permute(0, 3, 1, 2)  

        f2_mhsa = F.interpolate(f2_mhsa, size=(96, 96), mode='bilinear', align_corners=False)
        f12_feat = self.encoder_merge1_2(torch.cat([f1, f2_mhsa], dim=1)) 

        f3_reshaped = f3.permute(0, 2, 3, 1).reshape(batch_size, -1, 320) 
        f3_mhsa = self.famha_f3(f3_reshaped,f3_reshaped,f3_reshaped)
        f3_mhsa = f3_mhsa.reshape(batch_size, 24, 24, 320).permute(0, 3, 1, 2) 

        f4 = F.interpolate(f4, size=(24, 24), mode='bilinear', align_corners=False)
        f34_feat = self.encoder_merge3_4(torch.cat([f3_mhsa, f4], dim=1)) 
        
        f34_feat = F.interpolate(f34_feat, size=(96, 96), mode='bilinear', align_corners=False)
        f34_feat = self.trans_conv(f34_feat)  
        f1234_feat = self.encoder_merge1234(torch.cat([f12_feat, f34_feat], dim=1))

        conv4_cfrb_feats_upsampled = F.interpolate(conv4_cfrb_feats, size=(96, 96), mode='bilinear', align_corners=False)
        fused_feat = self.encoder_mergeall(torch.cat([f1234_feat, conv1_cfrb_feats, conv4_cfrb_feats_upsampled], dim=1)) 

        fused_feat = self.trans_conv(fused_feat)  
       
        fused_feat = F.interpolate(fused_feat, size=(112, 112), mode='bilinear', align_corners=False)
        fused_feat = self.dropout(fused_feat)  

        fused_feat = F.interpolate(fused_feat, size=(384, 384), mode='bilinear', align_corners=False)
        fused_feat = self.dropout(fused_feat)  

        output = self.ff_conv_1(fused_feat)
        return output


    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    weight_init(module)
