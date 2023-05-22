from torch import nn
import torch.nn.functional as F
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


#classes
###32*3300
class cnn2d(nn.Module):
    def __init__(self):
        super(cnn2d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = (5, 5), stride = (1,1), padding = (2,2)), ##1*15*16                
            nn.ReLU(),
            nn.MaxPool2d((2,2), (2,2))###16  1600
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 32, (5, 5), (1,1), (2,2)),
            nn.ReLU(),
            nn.MaxPool2d((2,2),(2,2)),   ####8 800     
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 128, (3, 3), (1,1), (1,1)),#128 8 800
            nn.ReLU(),
            nn.MaxPool2d((2,10),(2,10)), ###128 4 80
            nn.AdaptiveMaxPool2d((1,1))
        )
        
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32,1) #(32,1)不加这个会报错Target size (torch.Size([32, 1])) must be the same as input size (torch.Size([32, 2]))
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1,128)#全局池化后不加这个也会报错。
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x



"""
#classes
###32*3400
class cnn2d(nn.Module):
    def __init__(self):
        super(cnn2d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = (5, 5), stride = (1,1), padding = (2,2)), ##1*15*16                
            nn.ReLU(),
            nn.MaxPool2d((2,2), (2,2))###16  1600
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 32, (5, 5), (1,1), (2,2)),
            nn.ReLU(),
            nn.MaxPool2d((2,2),(2,2)),   ####8 800     
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), (1,1), (1,1)),#32 8 800
            nn.ReLU(),
            nn.MaxPool2d((2,10),(2,10)), ###32 4 80
        )
        
        self.fc1 = nn.Linear(64*4*80, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32,1) #(32,1)不加这个会报错Target size (torch.Size([32, 1])) must be the same as input size (torch.Size([32, 2]))
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1,64*4*80)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x



"""













'''    
#####72*4000
class cnn2d(nn.Module):
    def __init__(self):
        super(cnn2d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = (5, 5), stride = (1,1), padding = (2,2)), ##1*15*16                
            nn.ReLU(),
            nn.MaxPool2d((2,2), (2,2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 16, (5, 5), (1,1), (2,2)),
            nn.ReLU(),
            nn.MaxPool2d((2,2),(2,2)),        
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, (5, 5), (1,1), (2,2)),
            nn.ReLU(),
            nn.MaxPool2d((2,2),(2,2)), #32 9 500
        )
        self.conv4 = nn.Sequential(            
            nn.Conv2d(32, 64, (3, 5), (3,5), (0, 0)),##64 3 100
            nn.ReLU(),
            #nn.MaxPool2d((3,5), (3,5)),
            #nn.AdaptiveAvgPool2d((1,1))
        )
        
        self.fc1 = nn.Linear(64*3*100, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32,1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1,64*3*100)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

# net = cnn1d()
# print(net)




'''
