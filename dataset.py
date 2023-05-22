import torch
import pickle
from torch.utils.data import Dataset
from helps import read_pickle
    
class SHM_Dataset(Dataset):
    def __init__(self, path, mode):        
        self.data = read_pickle(path,f"{mode}_data")
        self.label = read_pickle(path,f"{mode}_label")
        #print(1)
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        # b,c,h,w = data.shape
        # data = data.reshape(b,c,1,-1)  
        #print(data)      
        #return torch.tensor(data).float().unsqueeze(0), torch.tensor(label).float().unsqueeze(0)#二分类
        return idx, torch.tensor(data).float().unsqueeze(0), torch.tensor(label)  #多分类 2d
        #return idx, torch.tensor(data).float().reshape(1,-1), torch.tensor(label)  #多分类 1d
        

    def __len__(self):
        return int(len(self.data)*1)  #int(len(self.data)*1)    #int(len(self.data)*0.5)  len(self.data)


class DI_Dataset(Dataset):
    def __init__(self, path, mode):        
        self.data = read_pickle(path,f"{mode}_data")
        self.label = read_pickle(path,f"{mode}_label")
        #print(1)
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]  
        #print(data)      
        return torch.tensor(data).float().unsqueeze(0), torch.tensor(label).float().unsqueeze(0)#二分类
        #return torch.tensor(data).float().unsqueeze(0), torch.tensor(label)  #多分类
       
    def __len__(self):
        return len(self.data)


class mlp_Dataset(Dataset):#mlp特征值数据 分类模型
    def __init__(self, data, label):
       self.data = data
       self.label = label

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        #return torch.tensor(data).float().unsqueeze(0), torch.tensor(label)  #多分类
        return torch.tensor(data).float().unsqueeze(0), torch.tensor(label).float().unsqueeze(0)#二分类

    def __len__(self):
        return len(self.data)

# if __name__=="__main__":
#     path = "/home/omnisky/data/4000/4000_pkl"#读取数据所在的文件夹
#     data = SHM_Dataset(path,"train")
    