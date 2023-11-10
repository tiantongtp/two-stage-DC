import os
import pickle
import random

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


#利用随机数种子，每次生成的随机数相同。   

def seed_torch(seed):#seed=torch.rand(42)  seed=42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)#在Numpy内部也有随机种子
    torch.manual_seed(seed) #为CPU中设置种子，生成随机数
    torch.cuda.manual_seed(seed)#为特定GPU设置种子，生成随机数
    torch.backends.cudnn.deterministic = True #后端  确定  为True的话，每次返回的卷积算法将是确定的，即默认算法。


def dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)



def remove_files(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))



