import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Res_AM(nn.Module):
    def __init__(self, c1, c2, s=2, k=3, r=16, c=4, d1=1, d2=2, d3=4):
        super().__init__()
        c_ = max(c1//r, c)
        self.cv0 = nn.Conv1d(c1, c_, kernel_size=1, stride=s)
        self.cv1 = nn.Conv1d(c_, c2, kernel_size=k, stride=1, padding=autopad(k=k,d=d1), dilation=d1)
        self.cv2 = nn.Conv1d(c_, c2, kernel_size=k, stride=1, padding=autopad(k=k,d=d2), dilation=d2)
        self.cv3 = nn.Conv1d(c_, c2, kernel_size=k, stride=1, padding=autopad(k=k,d=d3), dilation=d3)
        self.BR = nn.Sequential(
            nn.BatchNorm1d(3*c2),
            nn.ReLU()
        )
        self.cv = nn.Sequential() if c1==c2 and s==1 else nn.Conv1d(c1, c2, kernel_size=1, stride=s)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        c__ = max(3*c2//r, c)
        self.fc = nn.Sequential(
            nn.Linear(3*c2, c__, False),
            nn.BatchNorm1d(c__),
            nn.ReLU(),
            nn.Linear(c__, 3*c2, False),
            nn.BatchNorm1d(3*c2),
            nn.Sigmoid(),
        )
        
        
    def forward(self, x):
        x0 = self.cv0(x)
        x1 = self.cv1(x0)
        x2 = self.cv2(x0)
        x3 = self.cv3(x0)
        x0 = torch.cat((x1,x2,x3), 1)
        x0 = self.BR(x0)
        B, C, L = x0.size()
        avg = self.pool(x0).view([B, C])
        fc = self.fc(avg).view([B, C, 1])
        fc1, fc2, fc3 = torch.split(fc, C//3, dim=1)
        x1 = x1 * fc1
        x2 = x2 * fc2
        x3 = x3 * fc3
        return x1+x2+x3+self.cv(x)

class AMFCN(nn.Module):
    def __init__(self, c1=1, c2=5, device='cuda'):
        super().__init__()
        self.device = device
        self.conv = nn.Sequential(
            nn.Conv1d(c1, 16, kernel_size=513, stride=1, padding='same'),  #原始论文输入长度2048，k=1025
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.Res_AM1 = Res_AM(16, 32, s=2, k=13)
        self.Res_AM2 = Res_AM(32, 32, s=1, k=11)
        self.Res_AM3 = Res_AM(32, 64, s=2, k=9)
        self.Res_AM4 = Res_AM(64, 64, s=1, k=7)
        self.Res_AM5 = Res_AM(64, 128, s=2, k=5)
        self.Res_AM6 = Res_AM(128, 128, s=1, k=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, c2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.Res_AM1(x)
        x = self.Res_AM2(x)
        x = self.Res_AM3(x)
        x = self.Res_AM4(x)
        x = self.Res_AM5(x)
        x = self.Res_AM6(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x
        
if __name__ == "__main__":
    x = torch.randn(size=(64,1,1024)).to(device='cuda')
    # 对数据进行Random sampling
    B, C, L = x.size()
    RS = torch.randint(2, size=(B, C, L)).to(device='cuda')
    x = x * RS
    #
    test_model = AMFCN().to(device='cuda')

    output = test_model(x)
    print(f'输入尺寸为:{x.shape}')
    print(f'输出尺寸为:{output.shape}')
    # print(test_model)
    from torchsummary import summary 
    summary(test_model,(1,1024),device='cuda')  
        
        