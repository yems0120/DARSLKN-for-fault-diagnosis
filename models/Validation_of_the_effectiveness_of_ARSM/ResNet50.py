"""ResNet50 3-4-6-3"""

import torch
import torch.nn as nn

from modules.block import Conv

class Bottlrneck(torch.nn.Module):
    def __init__(self, In_channel, Med_channel, Out_channel, downsample=False):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            Conv(In_channel, Med_channel, 1, self.stride),
            Conv(Med_channel, Med_channel, 3),
            Conv(Med_channel, Out_channel, 1),
        )
        self.shortcut = Conv(In_channel, Out_channel, 1, self.stride) if In_channel != Out_channel or downsample else nn.Identity()  

    def forward(self,x):
        return self.layer(x) + self.shortcut(x)
    
    
class ResNet50(torch.nn.Module):
    def __init__(self,in_channels=1,classes=5):
        super(ResNet50, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 12, kernel_size=7, stride=2, padding=3),
            torch.nn.MaxPool1d(3,2,1),
            
            Bottlrneck(12,48,12, False),
            Bottlrneck(12,48,12, False),
            Bottlrneck(12,48,12, False),
            
            Bottlrneck(12,120,30, True),
            Bottlrneck(30,120,30, False),
            Bottlrneck(30,120,30, False),
            Bottlrneck(30,120,30, False),

            Bottlrneck(30,168,42, True),
            Bottlrneck(42,168,42, False),
            Bottlrneck(42,168,42, False),
            Bottlrneck(42,168,42, False),
            Bottlrneck(42,168,42, False),
            Bottlrneck(42,168,42, False),
            
            Bottlrneck(42,240,60, True),
            Bottlrneck(60,240,60, False),
            Bottlrneck(60,240,60, False),
            
            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(60,classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(-1, 60)
        x = self.classifer(x)
        return x
    
    
if __name__ == "__main__":
    x = torch.randn(size=(32,1,1024))
    test_model = ResNet50()
    output = test_model(x)
    print(f'输入尺寸为:{x.shape}')
    print(f'输出尺寸为:{output.shape}')
    from torchsummary import summary
    # summary(test_model,(1,1024),device='cuda')