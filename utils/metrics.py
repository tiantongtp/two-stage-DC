from tkinter import Y
import numpy as np
import torch
from sklearn.metrics import f1_score, average_precision_score,jaccard_score,precision_recall_curve, roc_auc_score,confusion_matrix,accuracy_score, precision_score,recall_score
from sklearn.metrics import roc_curve, auc
import warnings
import argparse
import configparser

from bunch import Bunch
from ruamel.yaml import safe_load
warnings.filterwarnings("ignore")
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
 

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return np.round(self.val, 4)  # decimals=4 表示小数点后四位

    @property
    def average(self):
        return np.round(self.avg, 4)




# # 多分类指标
def get_metrics(predict, target,threshold):
    
    _, predict_m = torch.max(predict.data, 1)
    target = target.cpu().detach().numpy().flatten()
    predict_m = predict_m.cpu().detach().numpy().flatten()
    cm = confusion_matrix(target,predict_m)   
    acc = accuracy_score(target,predict_m)  #multi label
    pre =precision_score(target,predict_m, average='macro')# macro micro
    sen =recall_score(target,predict_m, average='macro')
    f1 =f1_score(target,predict_m, average='macro')
    jc=jaccard_score(target,predict_m, average='macro')  

    return {
        
        "F1": np.round(f1, 4),
        "Acc": np.round(acc, 4),
        "Sen": np.round(sen, 4),       
        "pre": np.round(pre, 4),
        "jc": np.round(jc, 4),
        #"AUC": np.round(auc, 4),
        #"IOU": np.round(iou, 4),
    }





# # # 二分类指标
# def get_metrics(predict, target, threshold=0.5):
#     predict = torch.sigmoid(predict).cpu().detach().numpy()
#     predict_b = np.where(predict >= threshold, 1, 0)
#     if torch.is_tensor(target):
#         target = target.cpu().detach().numpy()
#     else:
#         target = target

#     tp = (predict_b * target).sum()
#     tn = ((1 - predict_b) * (1 - target)).sum()
#     fp = ((1 - target) * predict_b).sum()
#     fn = ((1 - predict_b) * target).sum()
#     auc = roc_auc_score(target, predict)  # ROC曲线下的面积
#     # precision, recall, thresholds = precision_recall_curve(target, output)
#     # precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
#     # recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
#     # pra = np.trapz(precision, recall)
#     # jc = jaccard_score(target, output_b)

#     acc = (tp + tn) / (tp + fp + fn + tn)
#     pre = tp / (tp + fp)
#     sen = tp / (tp + fn)  # = recall 灵敏度 召回率 真阳性率
#     spe = tn / (tn + fp)  # 特异度 真阴性率
#     iou = tp / (tp + fp + fn)
#     f1 = 2 * pre * sen / (pre + sen)
#     return {
#         "AUC": np.round(auc, 4),
#         "F1": np.round(f1, 4),
#         "Acc": np.round(acc, 4),
#         "Sen": np.round(sen, 4),
#         "Spe": np.round(spe, 4),
#         "pre": np.round(pre, 4),
#         "IOU": np.round(iou, 4),
#         # "PRA": np.round(pra, 4),
#         # "jc ": np.round(jc, 4),
#     }


