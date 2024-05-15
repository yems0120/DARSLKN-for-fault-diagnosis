import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# __all__ = ('CBAM', 'SENet', 'autopad', 'Conv', 'Bottleneck, 'C2f', 'ARNet')

# CBAM 注意力
class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Conv1d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))

class SpatialAttention(nn.Module):
    """Spatial-attention module."""
    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1))) 

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))
# CBAM 注意力


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

# ECA 注意力
class EfficientChannelAttention(nn.Module):
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x) # B*C*1
        fc = self.conv1(avg.transpose(-1, -2)).transpose(-1, -2)
        fc = self.sigmoid(fc)
        return x*fc
# ECA 注意力

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU() # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
    
    
class Bottleneck(nn.Module):  
    """DarknetBottleneck"""
    def __init__(self, c1, c2, shortcut=True, e=0.5,
                 k1=19, d1=2, k2=17, d2=4):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=k1, p=autopad(k=k1, d=d1), d=d1)
        self.cv2 = Conv(c_, c2, k=k2, p=autopad(k=k2, d=d2), d=d2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    

class C2f(nn.Module):  # CSPLayer_2Conv
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5,
                 k1=19, d1=2, k2=17, d2=4):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, e, k1, d1, k2, d2) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))   
        y.extend(m(y[-1]) for m in self.m)  
        return self.cv2(torch.cat(y, 1))  # Concat -> ConvModule
    
class ARNet(nn.Module): # Adaptive_Rank
    def __init__(self, channels, ratio=1):
        super(ARNet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//ratio, False),
            nn.SiLU(),
            nn.Linear(channels//ratio, channels, False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, L = x.size()
        avg = self.avg_pool(x).view([B, C])
        fc = self.fc(avg).view([B, C, 1])
        x = x*fc
        indices = torch.argsort(fc, dim=1, descending=False).expand(B, C, L)
        x = torch.gather(x, dim=1, index=indices)
        return x

# class ARNet(nn.Module):
#     def __init__(self, channels, b=1, gamma=2):
#         super(ARNet, self).__init__()
#         t = int(abs((math.log(channels, 2) + b) / gamma))
#         k = t if t % 2 else t + 1
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         B, C, L = x.size()
#         avg = self.avg_pool(x) # B*C*1
#         fc = self.conv1(avg.transpose(-1, -2)).transpose(-1, -2)
#         fc = self.sigmoid(fc)
#         x = x*fc
#         indices = torch.argsort(fc, dim=1, descending=False).expand(B, C, L)
#         x = torch.gather(x, dim=1, index=indices)
#         return x
   
   
class dilated_c2f1(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=1,
                 k1=19, d1=2, k2=17, d2=4):
        super().__init__()
        self.c = c1//2
        self.ar = ARNet(channels=c1)        
        self.cv = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, e, k1, d1, k2, d2) for _ in range(n))

    def forward(self, x):
        y = list(self.ar(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)  
        return self.cv(torch.cat(y, 1))
    
    
class dilated_c2f2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=1,
                 k1=19, d1=2, k2=17, d2=4):
        super().__init__()
        self.c = c1//3
        self.ar1 = ARNet(channels=c1)
        self.ar2 = ARNet(channels=(c1-self.c))         
        self.cv = Conv((5 + n) * self.c, c2, 1)
        self.m1 = nn.ModuleList(Bottleneck(2*self.c, 2*self.c, shortcut, e, k1, d1, k2, d2) for _ in range(n))
        self.m2 = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, e, k1, d1, k2, d2) for _ in range(n))

    def forward(self, x):
        y = list(torch.tensor_split(self.ar1(x), (self.c,), dim=1))
        y.extend(m1(y[-1]) for m1 in self.m1)
        x = self.ar2(y[-1])
        y.pop()
        y.extend(x.chunk(2, 1))
        y.extend(m2(y[-1]) for m2 in self.m2)
        return self.cv(torch.cat(y, 1))
    
class DR_CSP(nn.Module):
  def __init__(self):
    super().__init__()
    c = 48
    self.conv0 = Conv(1, c, k=1, s=1)

    self.conv1 = Conv(c, 2*c, k=3, s=2, p=1)
    self.c2f1 = dilated_c2f1(2*c, 2*c, n=1, shortcut=True, e=1,
                             k1=39, d1=2, k2=35, d2=5)
    
    self.conv2 = Conv(2*c, 3*c, k=3, s=2, p=1)
    self.c2f2 = dilated_c2f2(3*c, 3*c, n=1, shortcut=True, e=1,
                             k1=29, d1=2, k2=27, d2=5)
    
    self.pool = nn.AdaptiveAvgPool1d(1)
    self.drop = nn.Dropout(p=0.)
    self.fc = nn.Linear(3*c, 5)
  
  def forward(self, x):
    # x.shape: (batch, 1024, 1)
    x = self.conv0(x)
    
    x = self.conv1(x)
    x = self.c2f1(x)
    
    x = self.conv2(x)
    x = self.c2f2(x)
    
    x = self.pool(x).flatten(1)
    x = self.drop(x)
    x = self.fc(x) 
    return x


if __name__ == "__main__":
    x = torch.randn(size=(5,1,1024))
    test_model = DR_CSP()

    output = test_model(x)
    print(f'输入尺寸为:{x.shape}')
    print(f'输出尺寸为:{output.shape}')
    from torchsummary import summary
    summary(test_model,(1,1024),device='cuda')
    
    
""" 混淆矩阵
import torchmetrics  # TorchMetrics is a collection of 100+ PyTorch metrics implementations 
                    # and an easy-to-use API to create custom metrics.
model.to(device='cpu')
metric = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=5, normalize='none')
        '''
        normalize (Optional[Literal['true', 'pred', 'all', 'none']])
            None or 'none': no normalization (default)
            'true': normalization over the targets (most commonly used)
            'pred': normalization over the predictions
            'all': normalization over the whole matrix
        '''
for i, batch in enumerate(data_validate_loader):
    with torch.no_grad():
        sequence, targets = batch
        sequence = sequence.unsqueeze(1)
        outputs = model(sequence)
        cf = metric.update(outputs, targets)

fig_, ax_ = metric.plot()
metric.reset()
"""

"""
class dilated_c2f3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=1,
                 k1=19, d1=2, k2=17, d2=4):
        super().__init__()
        self.c = c1//4 # c
        self.ar1 = ARNet(channels=c1) # 4c
        self.ar2 = ARNet(channels=(c1-self.c)) # 3c
        self.ar3 = ARNet(channels=(c1-2*self.c)) # 2c
                 
        self.cv = Conv(10 * self.c, c2, 1)
        self.m1 = nn.ModuleList(Bottleneck(3*self.c, 3*self.c, shortcut, e, k1, d1, k2, d2) for _ in range(n))
        self.m2 = nn.ModuleList(Bottleneck(2*self.c, 2*self.c, shortcut, e, k1, d1, k2, d2) for _ in range(n))
        self.m3 = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, e, k1, d1, k2, d2) for _ in range(n))

    def forward(self, x):
        y = list(torch.tensor_split(self.ar1(x), (self.c,), dim=1))
        y.extend(m1(y[-1]) for m1 in self.m1)
        x1 = self.ar2(y[-1])
        y.pop()
        
        y.extend(torch.tensor_split(self.ar2(x1), (self.c,), dim=1)) 
        y.extend(m2(y[-1]) for m2 in self.m2)
        x2 = self.ar3(y[-1])
        y.pop()
        
        y.extend(x2.chunk(2, 1))
        y.extend(m3(y[-1]) for m3 in self.m3)
        
        return self.cv(torch.cat(y, 1))      
"""
