# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:43:17 2023

@author: lenovo
"""

'''
MSCNN-CRA
'''
import torch
from torch import nn
from functools import reduce
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
class CRattention(nn.Module):
    
    def __init__(self, channel, d):  
        super(CRattention, self).__init__()
        self.channel = channel
        self.d = d
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc=nn.ModuleList()  # 根据分支数量 添加 线性层      
        for i in range(d):
            self.fc.append(nn.Sequential(nn.Linear(self.channel, self.channel, bias=False)))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        b, c, l = x.size()
        y1 = self.avg_pool(x).view(b, c) 
        y2 = y1.reshape(b, self.channel, self.d)
        y3 = y2.permute(0,2,1)
        y4 = y3.reshape(b, c)
        attn =[]
        feature = []
        for i, fc in enumerate(self.fc):
            attn.append( fc(y4[:, self.channel*i:self.channel*(i+1)]) )
            feature.append(x[:,self.channel*i:self.channel*(i+1),:])
        #在第二个维度拼接
        U_re  = reduce(lambda x, y: torch.cat((x, y), dim=1), attn) #
        U_re  = U_re.reshape(b, self.d, self.channel)
        U_re  = U_re.permute(0,2,1)
        U_re  = U_re.reshape(b, c)
        U_re  = list(U_re.chunk(self.d, dim=1)) 
        U_re  = list(map(lambda x: self.softmax(x).reshape(b, self.channel, 1), U_re)) #
        
        U_s1 = list(map(lambda x,y: x * y.expand_as(x), feature, U_re))
        U_s1 = reduce(lambda x,y: x+y, U_s1)
        
        
        return U_s1

class SK1DConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=3):
        '''
        '''
        super(SK1DConv,self).__init__()
        

        self.M=M
        self.out_channels=out_channels
        self.conv1 = nn.Conv1d(in_channels*M, out_channels, 1, 1, 0, bias=False)
        self.conv=nn.ModuleList()    
        for i in range(M):
 
            k = 15
            self.conv.append(nn.Sequential(nn.Conv1d(in_channels,out_channels,k,stride,dilation= 1+i, padding= (k+(k-1)*i-1)//2, groups=2,bias=False),
                                           nn.ReLU(inplace=True),
                                           nn.BatchNorm1d(out_channels),
                                           ))
        self.crattn = CRattention(self.out_channels, self.M)
        # self.se = SELayer(out_channels)
        # self.multiHeadAttention = MultiHeadAttention(out_channels, out_channels//1,out_channels//1, 4)
    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        #the part of split
        for i,conv in enumerate(self.conv):
            output.append(conv(input)) 
        
        s1 = reduce(lambda x,y: torch.cat((x, y), dim=1), output)
        s2 = self.crattn(s1)
        return   s2
    
# x = torch.rand(5, 32, 128)
# model = SK1DConv(32, 32)
# y = model(x)
    
class SK1Block(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size, down_sample=False):
        super(SK1Block,self).__init__()
        
        self.down_sample = down_sample
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ReLU = nn.ReLU(inplace=True)
        stride = 1
        if down_sample:
            stride = 2  
            
        self.conv1=nn.Sequential(nn.Conv1d(in_channels, out_channels,1,1,0,bias=False),
                                 )
        self.BRC = nn.Sequential(
                                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                          padding=(kernel_size - 1) // 2),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(out_channels),
                                nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                                          padding=(kernel_size - 1) // 2),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(out_channels),                          
                               )  
        self.conv2=SK1DConv(out_channels, out_channels, 1)
        
        
        # self.flatten = nn.Flatten()
        self.average_pool = nn.AvgPool1d(kernel_size=1, stride=2)
        
    def forward(self, input):
        
        x = self.BRC(input)
        x = self.conv2(x)
        # print(x.size())
        
        if self.down_sample:  # 如果是下采样，则对输入进行平均池化下采样
            input = self.average_pool(input)
            
        if self.in_channels != self.out_channels:  # 如果输入的通道和输出的通道不一致，则进行padding,直接通过复制拼接矩阵进行padding,原代码是通过填充0
            # zero_padding = torch.zeros(input.shape).to(device)
            # input = torch.cat((input, zero_padding), dim=1)
            input = self.conv1(input)
        # print(input.size())
        result = x + input
        
        return result
    
class MBSDCN(nn.Module):
    def __init__(self,num_class=5):
        super(MBSDCN, self).__init__()
    
        self.num_class = num_class
        self.conv1 = nn.Sequential(
                                 nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
                                 # nn.BatchNorm1d(4),
                                 # nn.ReLU(inplace=True)
                                 )
                                    
        self.sklayer1 = SK1Block(in_channels=4,  out_channels= 8,   kernel_size=17, down_sample=True)
        self.sklayer2 = SK1Block(in_channels=8,  out_channels= 16,  kernel_size=17, down_sample=False)
        self.sklayer4 = SK1Block(in_channels=16, out_channels= 16,  kernel_size=17, down_sample=False)  # 8*256
        self.sklayer3 = SK1Block(in_channels=16, out_channels= 16,  kernel_size=17, down_sample=False)  # 8*256
        self.sklayer6 = SK1Block(in_channels=16, out_channels=16,   kernel_size=17, down_sample=False)  # 16*128
        # self.sklayer5 = SK1Block(in_channels=8, out_channels=16, kernel_size=3, down_sample=True) # 16*128
       
       
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()

        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
                                    nn.Linear(in_features=16, out_features=8),
                                    nn.BatchNorm1d(8),
                                    nn.ReLU(inplace=True),
                                    )
        self.output_class = nn.Sequential(
                                        nn.Linear(in_features=8, out_features=self.num_class),
                                        )
              

    def forward(self, input):  # 1*2048
        x = self.conv1(input)  # 4*1024
        x = self.sklayer1(x)  # 4*512
        x = self.sklayer2(x)  # 4*512
        x = self.sklayer4(x)  # 8*256
        x = self.sklayer3(x)  # 8*256
        # x = self.sklayer6(x)  # 16*128
        # print(x.size())
        x = self.bn(x)
        x = self.relu(x)
        gap = self.global_average_pool(x)  # 16*1

        # print( gap.size())
        gap = self.flatten(gap)  # 1*16
        feature = self.linear(gap)  # 1*8
        # feature = self.bn(feature)
        # feature  = self.relu(feature)
        output_class = self.output_class(feature)  # 1*3
       
        # return output_class, feature
        return output_class
 
if __name__ == '__main__':
    x = torch.rand(32, 1, 128)
    model = MBSDCN()
    y = model(x)
    print(y.size())
    from torchsummary import summary 
    summary(model,(1,1024),device='cuda') 
