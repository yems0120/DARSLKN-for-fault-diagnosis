"""AR_CSPNet18 2-2-2-2"""

import torch
import torch.nn as nn

from modules.block import Conv, ARNet

class Bottleneck(nn.Module):  
    def __init__(self, channel):
        super().__init__()
        self.cv1 = Conv(channel, channel, 3, p=1)
        self.cv2 = Conv(channel, channel, 3, p=1)  

    def forward(self, x):
        return self.cv2(self.cv1(x))
    
    
class AR_CSPBlock(nn.Module):
    def __init__(self, c1, c2, n=1, dowsample=True):
        super(AR_CSPBlock, self).__init__()    
        self.strde = 2
        if dowsample == False:
            self.strde = 1
            
        self.n = n
        self.c_ = int(c1/(n+1))
        self.m = nn.ModuleList(Bottleneck(c1-self.c_*(i+1)) for i in range(n))
        self.ar = nn.ModuleList(ARNet(c1-self.c_*j) for j in range(n))
        self.conv = Conv(int((n+1)*(n+2)*self.c_/2), c2, 1, s=self.strde)
        
    def forward(self, x):
        y = []
        for brach in range(self.n):
            x = self.ar[brach](x)
            y.append(x)
            y1, y2 = torch.tensor_split(x, (self.c_,), dim=1)
            x = self.m[brach](y2)
        y.append(x)
        return self.conv(torch.cat(y, dim=1))
    
    
class AR_CSP18(nn.Module):
    def __init__(self, in_channels=1, classes=5):
        super(AR_CSP18, self).__init__()
        self.stem = nn.Sequential(
            Conv(in_channels, 12, k=7, s=2, p=3, act=True),
            torch.nn.MaxPool1d(3,2,1),
            )
        self.csp1 = AR_CSPBlock(12, 30, n=2)
        self.csp2 = AR_CSPBlock(30, 42, n=2)
        self.csp3 = AR_CSPBlock(42, 60, n=2)
        self.csp4 = AR_CSPBlock(60, 60, n=2, dowsample=False)        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(60, classes)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.csp1(x)
        x = self.csp2(x)
        x = self.csp3(x)
        x = self.csp4(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x
    
    
if __name__ == "__main__":
    x = torch.randn(size=(32,1,1024))
    test_model = AR_CSP18()
    output = test_model(x)
    print(f'输入尺寸为:{x.shape}')
    print(f'输出尺寸为:{output.shape}')
    from torchsummary import summary
    summary(test_model,(1,1024),device='cuda')