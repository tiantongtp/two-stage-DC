import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import sys
import torch.nn.init as init

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, padding=1, bias=True)
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)






class CNN1D(nn.Module):
    def __init__(self, depth, in_planes, planes, num_classes, stride=1,dropout=0):
        super(CNN1D, self).__init__()
        self.planes = 64

        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=100, stride=stride, bias=True)
        self.bn1 = nn.BatchNorm1d(planes) 
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=100, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm1d(planes) 
        self.conv3 = nn.Conv1d(planes, planes, kernel_size=50, bias=True)
        self.bn3 = nn.BatchNorm1d(planes)
        self.maxpool = nn.MaxPool1d(kernel_size=20, stride=2, padding=1) 
        self.linear = nn.Linear(planes, num_classes)    
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.linear = nn.Linear(planes, num_classes)
     



    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn1(self.conv1(x.squeeze(1))))
        out = self.maxpool(out)
        out = self.maxpool(F.relu(self.bn2(self.conv2(out))))
        out = self.maxpool(F.relu(self.bn3(self.conv3(out))))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        #out=self.linear3(self.dp((self.linear2(self.linear1(out)))))
        #out=self.linear2(self.dp((self.linear1(out))))
        return out
