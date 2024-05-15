import torch
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, in_channel=1, classes=5):
        super().__init__()
        
        self.cv1 = nn.Sequential(nn.Conv1d(in_channel, 128, 8, padding='same'),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU())
        self.cv2 = nn.Sequential(nn.Conv1d(128, 256, 5, padding=2),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU())
        self.cv3 = nn.Sequential(nn.Conv1d(256, 128, 3, padding=1),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 5)
    def forward(self, x):
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.cv3(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x
    
if __name__=='__main__':
    x = torch.randn(size=(64,1,1024))
    test_model = FCN()
    output = test_model(x)
    print(f'输入尺寸为:{x.shape}')
    print(f'输出尺寸为：{output.shape}')
     