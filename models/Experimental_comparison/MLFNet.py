import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, False),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels, False), 
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, l = x.size()  # batch, channel, length
        x1 = self.avg_pool(x).view(b, c)
        avg_out = self.fc(x1).view(b, c, 1)
        x2 = self.max_pool(x).view(b, c)
        max_out = self.fc(x2).view(b, c, 1)
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x
    

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1 if stride!=1 else 'same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=16, stride=stride, padding=7 if stride!=1 else 'same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=64, stride=stride, padding=31 if stride!=1 else 'same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return x1 + x2 + x3
    

class MLFNet(nn.Module):
    def __init__(self, med_channels=16, input_channels=1, num_classes=5):
        super(MLFNet, self).__init__()
        self.level1 = Bottleneck(input_channels, med_channels, 4)
        self.level2 = Bottleneck(med_channels, med_channels, 1)
        self.level3 = Bottleneck(med_channels, med_channels, 1)
        self.attention = ChannelAttention(med_channels*3)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.fc = nn.Sequential(
            nn.Linear(med_channels*3*64, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
            nn.Softmax(),
        )
        
    def forward(self, x):
        x1 = self.level1(x)
        x2 = self.level2(x1)
        x3 = self.level3(x2)
        x = self.attention(torch.cat((x1, x2, x3), dim=1))
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x        


if __name__ == '__main__':
    test_model = MLFNet(med_channels=48)
    x = torch.randn(size=(5,1,1024))
    output = test_model(x)
    print(f'输入尺寸为:{x.shape}')
    print(f'输出尺寸为:{output.shape}')
    # summary(test_model, (1, 1024), device='cpu')