import argparse
import os
import torch.nn as nn
import torch
from bunch import Bunch
from loguru import logger
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
from torchstat import stat
import models
import wandb
from dataset import mlp_Dataset
from trainer import Trainer
from utils import losses
from utils.helpers import get_instance, seed_torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
def train(CFG):
    seed_torch(42)
    names=['name','di_max','di_sst','di_sss','di_sdcc','label']#给每列命名
    data_orginal=pd.read_csv(CFG["data_path"],names=names)#导入原始CSV文件   
    data=np.array(data_orginal[['di_max','di_sst','di_sss','di_sdcc']])#取特征x
    label= np.array(data_orginal['label'])#0 1标签y
    train_data,test_data,train_label,test_label =train_test_split(data,label, random_state=1, train_size=0.7,test_size=0.3)
    train_dataset = mlp_Dataset(train_data, train_label)
    test_dataset = mlp_Dataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, CFG.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, CFG.batch_size, shuffle=False,  num_workers=16, pin_memory=True)

    logger.info('The patch number of train is %d' % len(train_dataset))#比print更方便

    model = get_instance(models, 'model', CFG)
    # stat(model, (1, 48, 48))
    logger.info(f'\n{model}\n')
    # LOSS
    loss = get_instance(losses, 'loss', CFG)
    # TRAINING
    trainer = Trainer(
        model=model,
        mode="train",
        loss=loss,
        CFG=CFG,
        train_loader=train_loader,
        test_loader=test_loader
    )

    return trainer.train()
def test(save_path, checkpoint_name, CFG, show):
    checkpoint = torch.load(os.path.join(save_path, checkpoint_name))
    CFG_ck = checkpoint['config']
    # DATA LOADERS
    test_dataset = mlp_Dataset(**CFG["data_set"], mode="test")
    test_loader = DataLoader(test_dataset, CFG.batch_size,
                             shuffle=False,  num_workers=16, pin_memory=True)
    # MODEL
    model = get_instance(models, 'model', CFG_ck)
    # LOSS
    loss = get_instance(losses, 'loss', CFG_ck)
    
    # TEST
    tester = Trainer(model=model, mode="test", loss=loss, CFG=CFG, checkpoint=checkpoint,
                     test_loader=test_loader, save_path=save_path,show=show)
    tester.test(metrics_type="TTA",)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.yaml', type=str,
                        help='Path to the config file (default: config.yaml)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-n', '--name', default=None, type=str,
                        help='the wandb name of run')
    args = parser.parse_args()

    # yaml = YAML(typ='safe')
    with open('/home/omnisky/code/Damage_prediction/config.yaml', encoding='utf-8') as file: 
        CFG = Bunch(safe_load(file))  # 为列表类型
    os.environ["CUDA_VISIBLE_DEVICES"] = CFG["CUDA_VISIBLE_DEVICES"]
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
     
    wandb.init(project="ECC", config=CFG,
               sync_tensorboard=True, name=f" {CFG['model']['type']} {CFG['loss']['type']} {args.name}")
    path = train(CFG)
    checkpoint_name = "best_model.pth"
    # test(path, checkpoint_name, CFG,show = True)
