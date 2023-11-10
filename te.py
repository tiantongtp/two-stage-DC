from helps import read_pickle
import numpy as np
path='/home/tiantong/data/tt/datanew/pkldata/classify/newpre/73/4000/c375'
test_label = read_pickle(path,'test_label')

np.set_printoptions(threshold  =  1e6 )
b=len(test_label)
for i in range(0, len(test_label)):
    with open('./c375testlabel.txt', 'a') as f:  # 设置文件对象
        
        print(test_label[i],file = f)

# print(test_label)