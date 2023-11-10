from torch import nn
import torch.nn.functional as F
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)
class cnn1d(nn.Module):
    def __init__(self, num_classes):
        super(cnn1d, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=100,  stride=1, bias=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.mp1 = nn.MaxPool1d(kernel_size=20, stride=2, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=100, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm1d(64)
        self.mp2 = nn.MaxPool1d(kernel_size=20, stride=2, padding=1) 
        self.conv3 = nn.Conv1d(64, 64, kernel_size=50, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm1d(64)
        self.mp3 = nn.MaxPool1d(kernel_size=20, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.linear = nn.Linear(64, num_classes)
        
    def forward(self, x):
        out = self.mp1(F.relu(self.bn1(self.conv1(x.squeeze(1)))))
        out = self.mp2(F.relu(self.bn2(self.conv2(out))))
        out = self.mp3(F.relu(self.bn3(self.conv3(out))))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
      
        return out



