import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()


class CFRB(nn.Module):
    def __init__(self, feature_layer=None, out_channels=64):
        super(CFRB, self).__init__()

        self.dil_rates = [3, 5, 7, 9]

        if feature_layer == '1':
            self.in_channels = 64
        elif feature_layer == '2':
            self.in_channels = 128
        elif feature_layer == '3':
            self.in_channels = 320
        elif feature_layer == '4':
            self.in_channels = 512

        # Define layers
        self.conv_1_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.conv_dil_3 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[0], padding=self.dil_rates[0], bias=False)
        self.conv_dil_5 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[1], padding=self.dil_rates[1], bias=False)
        self.conv_dil_7 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[2], padding=self.dil_rates[2], bias=False)
        self.conv_dil_9 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[3], padding=self.dil_rates[3], bias=False)
        
    
        self.bn = nn.BatchNorm2d(out_channels*5)

    def forward(self, input_):
        # Extract features
        conv_1_1_feats = self.conv_1_1(input_)
        conv_dil_3_feats = self.conv_dil_3(input_)
        conv_dil_5_feats = self.conv_dil_5(input_)
        conv_dil_7_feats = self.conv_dil_7(input_)
        conv_dil_9_feats = self.conv_dil_9(input_)

        # Aggregate features
        concat_feats = torch.cat((conv_1_1_feats, conv_dil_3_feats, conv_dil_5_feats, conv_dil_7_feats, conv_dil_9_feats), dim=1)
        bn_feats = F.relu(self.bn(concat_feats))

        return bn_feats

    def initialize(self):
        weight_init(self)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
    
    def initialize(self):
        # Initialize only if layers have parameters
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight_init(module)

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
    
    def initialize(self):
        weight_init(self)

# Adaptive Spatial Coordinate Attention (ACA)
class ASCA(nn.Module):
    def __init__(self, inp=64, oup=64, reduction=32, groups=4):
        super(ASCA, self).__init__()
        
        # Ensure mip is divisible by groups
        mip = max(8, (inp // reduction // groups) * groups)
        
        # Global pooling
        self.pool_h = lambda x: x.mean(dim=-1, keepdim=True)
        self.pool_w = lambda x: x.mean(dim=-2, keepdim=True)
        
        # Grouped convolution
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, groups=groups)
        self.bn1 = nn.GroupNorm(num_groups=groups, num_channels=mip)
        self.act = nn.SiLU(inplace=True)
        
        # Final transformations
        self.fc_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, groups=groups)
        self.fc_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, groups=groups)
    
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # Pooling and processing
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        # Combine and process
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split and transform
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = torch.sigmoid(self.fc_h(x_h))
        a_w = torch.sigmoid(self.fc_w(x_w))
        
        return identity * a_h * a_w

    
    def initialize(self):
        weight_init(self)


# Compact Channel Gate
class CCG(nn.Module):

    def __init__(self, max_kernel=5):
        super().__init__()
        self.max_kernel = max_kernel
        self.sigmoid = nn.Hardsigmoid()  # Quantization-friendly
        
    def forward(self, x):
        C = x.size(1)
        kernel_size = min(max(3, C // 32), self.max_kernel)  # Dynamic kernel size
        padding = (kernel_size - 1) // 2

        # Simplified GAP
        y = x.mean(dim=(-2, -1), keepdim=True)  # Spatial mean
        y = y.squeeze(-1).permute(0, 2, 1)  # [bs, 1, c]

        # Depthwise 1D convolution
        y = F.conv1d(y, weight=torch.ones(1, 1, kernel_size, device=x.device) / kernel_size, padding=padding)
        y = self.sigmoid(y)  # Apply activation

        y = y.permute(0, 2, 1).unsqueeze(-1)  # Reshape back
        return x * y.expand_as(x)  # Channel-wise modulation


    def initialize(self):
        # Initialize only if layers have parameters
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight_init(module)


# Feature-Aware Multi-Head Attention
class FAMHA(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, H=7, W=7, ratio=2, apply_transform=True):
        super(FAMHA, self).__init__()

        self.H = H
        self.W = W
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.ratio = ratio
        if self.ratio > 1:
            self.sr = nn.Sequential()
            self.sr_conv = nn.Conv2d(d_model, d_model, kernel_size=1, stride=2, padding=0, groups=d_model)
            self.sr_ln = nn.LayerNorm(d_model)

        self.apply_transform = apply_transform and h > 1
        if self.apply_transform:
            self.transform = nn.Sequential(
                nn.Conv2d(h, h, kernel_size=1, stride=1),
                nn.Softmax(dim=-1)
            )

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)  # Lightweight initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq, c = queries.shape
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)

        if self.ratio > 1:
            x = queries.permute(0, 2, 1).view(b_s, c, self.H, self.W)  # bs, c, H, W
            x = self.sr_conv(x)  # bs, c, h, w
            x = x.contiguous().view(b_s, c, -1).permute(0, 2, 1)  # bs, n', c
            x = self.sr_ln(x)
            k = self.fc_k(x).view(b_s, -1, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, n')
            v = self.fc_v(x).view(b_s, -1, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, n', d_v)
        else:
            k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
            v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
        if self.apply_transform:
            att = self.transform(att)  # (b_s, h, nq, n')
        else:
            att = torch.softmax(att, -1)  # (b_s, h, nq, n')

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out



    def initialize(self):
        # Initialize only if layers have parameters
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight_init(module)
 


 
