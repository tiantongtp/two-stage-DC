import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from helps import dir_exists
snr=25
def awgn(x, snr):
    '''
    加入高斯白噪声 
    :x: 原始信号
    :snr: 信噪比  值越小噪声越大
    '''
    height = len(x)
    width = len(x[0])
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / (len(x)*len(x[0]))
    npower = xpower / snr
    noise = np.random.normal(size=[height, width]) * np.sqrt(npower)
    return x + noise
    
def data_preprocess(data_path,save_path):

    dir_exists(save_path)
    data_id = os.path.join(label_path,"labelDall.csv")#数据文件夹中有label文件。#标签  label4 labelstringer1 labelskinback1 labelskinfront1 labelall 
    label_file = pd.read_csv(data_id,header=None)
    data_ls = []

    for id,i in  enumerate(label_file[0]):
        data = np.array(pd.read_csv(os.path.join(data_path,i),header=None))#[:,600:]  #3400
        data[:,0:599]=0 #将前400行替换为0 4000       
        #mn = data.mean()
        #std = data.std()
        #print(data[c].shape, data[c].dtype, mn, std)        
        #data = np.exp(data)
        #data = (data - data.mean()) / (data.max() - data.min())#mean normalization
        
        #data = awgn(data,snr) #加白噪声     
        data_ls.append(data)
        print(f"{id}:{i}  {label_file[1][id]}")

    label=np.array(label_file[1])
    train_data,test_data,train_label,test_label =train_test_split(data_ls,label, random_state=1,
                                                train_size=0.7,test_size=0.3)
                                                  #stratify=label,分层划分数据集
    save_data(train_data,save_path,"train_data")
    save_data(train_label,save_path,"train_label")
    save_data(test_data,save_path,"test_data")
    save_data(test_label,save_path,"test_label")

def save_data(data_list, path, type):
    with open(file=os.path.join(path, f'{type}.pkl'), mode='wb') as file:
        pickle.dump(data_list, file)
        print(f'save {type} : {type}.pkl')


if __name__ == "__main__":
    data_path = "/home/tiantong/data/tt/datanew/csvdata100k"#读取数据所在的文件夹 _snr30 _snr25 _snr20
    label_path = "/home/tiantong/data/tt/datanew/label"
    save_path = "/home/tiantong/data/tt/datanew/pkldata/classify/newpre/73/4000/c375"  # /newpre/73/4000/stratify/c639_snr25  /OR/73/4000/stratify/c375
    data_preprocess(data_path,save_path)