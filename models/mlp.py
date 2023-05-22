from torch import nn
import torch.nn.functional as F
import torch
import numpy

class mlp4(torch.nn.Module):##四个特征在一个csv文件中
    def __init__(self):
        super(mlp4,self).__init__()
        self.mlp = torch.nn.Sequential(

            nn.Linear(1,1),  # 输入层与第一隐层结点数设置，全连接结构
            torch.nn.Sigmoid(),  # 第一隐层激活函数采用sigmoid
            # nn.Linear(256,128),  # 第一隐层与第二隐层结点数设置，全连接结构
            # #torch.nn.Sigmoid(),  # 第一隐层激活函数采用sigmoid
            # nn.Linear(128,32),  # 第二隐层与输出层层结点数设置，全连接结构
            # #torch.nn.Sigmoid(),  # 第一隐层激活函数采用sigmoid
            # nn.Linear(32,1),  # 第二隐层与输出层层结点数设置，全连接结构
            
        )
        self.fc1=nn.Linear()
    def forward(self, x):
        #x = x.flatten()
        x = self.mlp(x)
        x = x.view(-1,1)      
        return x
# '''

# class mlp4(torch.nn.Module):##四个特征在一个csv文件中
#     def __init__(self):
#         super(mlp4,self).__init__()
#         self.mlp = torch.nn.Sequential(

#             nn.Linear(4,32),  # 输入层与第一隐层结点数设置，全连接结构
#             torch.nn.Sigmoid(),  # 第一隐层激活函数采用sigmoid
#             nn.Linear(32,64),  # 第一隐层与第二隐层结点数设置，全连接结构
#             torch.nn.Sigmoid(),  # 第一隐层激活函数采用sigmoid
#             nn.Linear(64,32),  # 第二隐层与输出层层结点数设置，全连接结构
#             torch.nn.Sigmoid(),  # 第一隐层激活函数采用sigmoid
#             nn.Linear(32,7),  # 第二隐层与输出层层结点数设置，全连接结构
#         )
#     def forward(self, x):
#         x = self.mlp(x)
#         x = x.view(-1,7)        
#         return x
# '''
