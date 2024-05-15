import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class MC_CNN(nn.Module):
    def __init__(self, c1=1, c2=5):
        super().__init__()
        self.BN = nn.BatchNorm1d(c1)
        self.sc1 = nn.Sequential(
            nn.Conv1d(c1, c1, kernel_size=100),
            nn.BatchNorm1d(c1),
            # nn.Sigmoid()
            nn.ReLU()   
        )
        self.sc2 = nn.Sequential(
            nn.Conv1d(c1, c1, kernel_size=200),
            nn.BatchNorm1d(c1),
            # nn.Sigmoid()
            nn.ReLU()   
        )
        self.sc3 = nn.Sequential(
            nn.Conv1d(c1, c1, kernel_size=300),
            nn.BatchNorm1d(c1),
            # nn.Sigmoid()   
            nn.ReLU() 
        )
        self.cv1 = nn.Sequential(
            nn.Conv1d(c1, 32, kernel_size=8, stride=2),
            nn.BatchNorm1d(32),
            # nn.Sigmoid()  
            nn.ReLU()  
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.cv2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=32, stride=4),
            nn.BatchNorm1d(32),
            # nn.Sigmoid()  
            nn.ReLU()  
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.cv3 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=16, stride=2),
            nn.BatchNorm1d(32),
            # nn.Sigmoid()
            nn.ReLU() 
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.drop = nn.Dropout(p=0.4)
        self.fc1 = nn.Sequential(
            nn.Linear(448, 112),
            nn.BatchNorm1d(112),
            # nn.Sigmoid(),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(112, c2)   
        
    def forward(self, x):
        # 对x归一化
        self.BN(x)
        #
        x1 = self.sc1(x)
        x2 = self.sc2(x)
        x3 = self.sc3(x)
        x = torch.concat((x1,x2,x3), 2)
        x = self.cv1(x)
        x = self.pool1(x)
        x = self.cv2(x)
        x = self.pool2(x)
        x = self.cv3(x)
        x = self.pool3(x).flatten(1)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    x = torch.randn(size=(64,1,1024))
    test_model = MC_CNN()

    output = test_model(x)
    print(f'输入尺寸为:{x.shape}')
    print(f'输出尺寸为:{output.shape}')        