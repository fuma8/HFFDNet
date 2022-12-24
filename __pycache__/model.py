import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init


class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, n_feats, groups=4):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, n_feats // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, n_feats // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, n_feats // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, n_feats // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(n_feats // (2 * groups), n_feats // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out

def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=32, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    temp = torch.nn.PixelShuffle(32)(temp)
    return temp


class HFEU(nn.Module):
    def __init__(self, channels):
        super(HFEU, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(self.channels, self.channels, 3, padding = 1, bias = False)
        self.refine = nn.Conv2d(self.channels, self.channels, 5, padding = 2, bias = False)
        self.act = nn.ReLU(inplace = True)
    def forward(self, x):
        x_h = self.act(self.conv1(x))
        x_res = x_h - x
        x_refine = self.act(self.refine(x_res))
        x_output = x_refine + x
        return x_output

class FFDB(nn.Module):
    def __init__(self, channels, distillation_rate = 0.5):
        super(FFDB, self).__init__()
        self.channels = channels
        self.distilled_channels = int(self.channels*distillation_rate)
        self.remaining_channels = int(self.channels  - self.distilled_channels)
        self.HFEU1 = HFEU(self.remaining_channels)
        self.HFEU2 = HFEU(self.remaining_channels)
        self.HFEU3 = HFEU(self.remaining_channels)
        self.conv = nn.Conv2d(self.channels, self.channels, 3, padding = 1, bias = False)
        self.conv1 = nn.Conv2d(self.remaining_channels, self.channels, 1, bias = False)
        self.conv2 = nn.Conv2d(self.remaining_channels, self.channels, 1, bias = False)
        self.conv3 = nn.Conv2d(self.remaining_channels, self.distilled_channels, 1, bias = False)
        self.c5 = nn.Conv2d(self.distilled_channels*4, self.channels, 1, bias = False)
        self.sa = sa_layer(self.channels)

    def forward(self, x):
        x_input = self.conv(x)
        distilled_input, remaining_input = torch.split(x_input, (self.distilled_channels, self.remaining_channels), dim = 1)
        h1 = self.HFEU1(remaining_input)
        r1 = self.conv1(h1)
        distilled_input2, remaining_input2 = torch.split(r1, (self.distilled_channels, self.remaining_channels), dim = 1)
        h2 = self.HFEU2(remaining_input2)
        r2 = self.conv2(h2)
        distilled_input3, remaining_input3 = torch.split(r2, (self.distilled_channels, self.remaining_channels), dim = 1)
        h3 = self.HFEU3(remaining_input3)
        r3 = self.conv3(h3)
        x_con = torch.cat([distilled_input, distilled_input2, distilled_input3, r3], dim = 1)
        x_esa = self.sa(self.c5(x_con))
        return x_esa + x

class ResidualBlock(nn.Module):
    def __init__(self, nf, kz, bias):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, kz, padding = kz // 2, bias = bias), nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kz, padding=kz // 2, bias = bias)
        )
    
    def forward(self, x):
        return x + self.body(x)

class Phase(nn.Module):
    def __init__(self, channels = 32):
        super(Phase, self).__init__()
        self.rho = nn.Parameter(torch.Tensor([0.5]))
        self.channels = channels
        self.conv1 = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels+1, 3, 3)))
        self.RB1 = ResidualBlock(self.channels, 3, bias = True)
        self.RB2 = ResidualBlock(self.channels, 3, bias = True)
        self.conv2 = nn.Parameter(init.xavier_normal_(torch.Tensor(1, self.channels, 3, 3)))
        self.FFDB = FFDB(self.channels)
    
    def forward(self, x, z, PhiWeight, PhiTWeight, PhiTb):
        x = x - self.rho * PhiTPhi_fun(x, PhiWeight, PhiTWeight)
        x_input = x + self.rho * PhiTb
        x_a = torch.cat([x_input, z], 1)
        x_D = F.conv2d(x_a, self.conv1, padding=1)
        x_R = self.RB1(x_D)
        # write distillation based super resolution code
        x_FFDB = self.FFDB(x_R)
        x_backward = self.RB2(x_FFDB)
        x_G = F.conv2d(x_backward, self.conv2, padding=1)
        x_pred = x_input + x_G
        return x_pred, x_backward

class HFFDNet(nn.Module):
    def __init__(self, sensing_rate, LayerNo, channels = 32):
        super(HFFDNet, self).__init__()
        self.measurement = int(sensing_rate * 1024)
        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(self.measurement, 1024)))
        self.LayerNo = LayerNo
        self.channels = channels
        layer = []
        for i in range(self.LayerNo):
            layer.append(Phase())

        self.fcs = nn.ModuleList(layer)
        self.fe = nn.Conv2d(1, channels, 3, padding=1, bias=True)

    def forward(self, x):
        PhiWeight = self.Phi.contiguous().view(self.measurement, 1, 32, 32) #Phi
        PhiTWeight = self.Phi.t().contiguous().view(1024, self.measurement, 1, 1) #PhiT
        y = F.conv2d(x, PhiWeight, padding = 0, stride = 32, bias = None) #y
        PhiTb = F.conv2d(y, PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(32)(PhiTb) # phiTy
        x = PhiTb
        z = self.fe(x)

        for i in range(self.LayerNo):
            x, z = self.fcs[i](x, z, PhiWeight, PhiTWeight, PhiTb)
        
        x_final = x
        phi_cons = torch.mm(self.Phi, self.Phi.t())

        return x_final, phi_cons