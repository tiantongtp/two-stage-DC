from re import X
from sklearn.svm import SVC
from torch import nn
import torch.nn.functional as F
import torch
# from sklearn.linear_model import LogisticRegression#逻辑回归：
# from sklearn.naive_bayes import GaussianNB#朴素贝叶斯：
# from sklearn.neighbors import KNeighborsClassifier#K-近邻：
# from sklearn.tree import DecisionTreeClassifier#决策树：


#clf = SVC(C=0.1, kernel='linear', decision_function_shape='ovo')
class svm(nn.Module):
    def __init__(self):
        super(svm, self).__init__()
        clf = SVC(C=0.1, kernel='linear', decision_function_shape='ovo') # One in and one out
        self.linear = clf.fit()

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1,1)  
        return X