import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# SE 注意力
class SENet(nn.Module):
    def __init__(self, channels, ratio=16):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//ratio, False),
            nn.ReLU(),
            nn.Linear(channels//ratio, channels, False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        N, c, L = x.size()
        avg = self.avg_pool(x).view([N, c])
        fc = self.fc(avg).view([N, c, 1])

        return x*fc
# SE 注意力

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class RCB(nn.Module):
    def __init__(self, in_channel=1):
        super().__init__()    
        self.cv1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=64, stride=8, padding=31),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.poo11 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.cv2 = nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=3)
        
        self.poo12 = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.cv = nn.Conv1d(in_channel, 32, kernel_size=1, stride=16)
      
    def forward(self, x):
        x1 = self.cv1(x)
        x1 = self.poo11(x1)
        x1 = self.cv2(x1)
        x = 0.2*self.cv(x)+x1
        return self.poo12(x)


class DRCB(nn.Module):
    def __init__(self,):
        super().__init__()       
        self.dcv1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.dcv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=autopad(k=3,d=2), dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        ) 
        self.dcv3 = nn.Conv1d(64, 64, kernel_size=3, padding=autopad(k=3,d=3), dilation=3)
        self.BR = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.cv = nn.Conv1d(32, 64, kernel_size=1)
        
    def forward(self, x):
        x2 = self.dcv1(x)
        x2 = self.dcv2(x2)
        x2 = self.dcv3(x2)
        x = 0.2*self.cv(x)+x2
        return self.BR(x)
    
class DCNN_RC(nn.Module):
    def __init__(self, in_channel=1):
        super().__init__()
        self.RCB = RCB(in_channel=in_channel)
        self.DRCB = DRCB()
        self.se1 = SENet(32, ratio=4)
        self.se2 = SENet(64, ratio=4)
        # self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.cv1 = nn.Conv1d(1, 64, kernel_size=1, stride=32)
        self.cv2 = nn.Conv1d(32, 64, kernel_size=1)
        
        self.drop = nn.Dropout(p=0.5)
        
        self.fc = nn.Sequential(
            nn.Linear(16*64, 100),
            nn.ReLU(),
            nn.Linear(100, 5)
        )
        
        
    def forward(self, x):
        x1 = self.RCB(x)
        x1 = self.se1(x1)
        x2 = self.DRCB(x1)
        x = 0.2*self.cv1(x) + 0.2*self.cv2(x1)+x2
        x = self.se2(x)
        x = self.pool3(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)
        
if __name__ == "__main__":
    x = torch.randn(size=(64,1,1024))
    test_model = DCNN_RC()

    output = test_model(x)
    print(f'输入尺寸为:{x.shape}')
    print(f'输出尺寸为:{output.shape}')
    # print(test_model)
    # summary(test_model,(1,1024),device='cuda')
    # dropout = 0.5 
    from torchsummary import summary 
    summary(test_model,(1,1024),device='cuda')     
        
        