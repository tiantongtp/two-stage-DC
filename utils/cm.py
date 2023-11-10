import torch
from sklearn.metrics import  confusion_matrix 
import matplotlib.pyplot as plt





# 显示混淆矩阵
def plot_confuse_data(predict, target, threshold):
    classes = range(0,4)
    _, predict_m = torch.max(predict.data, 1)
    target = target.cpu().detach().numpy().flatten()
    predict_m = predict_m.cpu().detach().numpy().flatten()
    cm = confusion_matrix(target,predict_m)  
    plt.imshow(cm, cmap=plt.cm.Blues) #颜色风格为蓝色

    indices = range(len(cm))
     # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion matrix')

 # 显示数据
    for first_index in range(len(cm)):    #第几行
        for second_index in range(len(cm[first_index])):    #第几列
            plt.text(first_index, second_index, cm[first_index][second_index])
 # 显示
    plt.show()
    plt.save('/home/tiantong/code/img_save/cm.jpg')


