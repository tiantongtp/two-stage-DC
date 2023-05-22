import argparse
import os
import torch.nn as nn
import torch
from bunch import Bunch
from loguru import logger
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
from torchstat import stat
import random
import models
import wandb
from dataset import SHM_Dataset, DI_Dataset
from trainer import Trainer
from utils import losses
from utils.helpers import get_instance, seed_torch



def train(CFG):
    if CFG.seed == "RANDOM":
        seed = random.randint(0,10000000)
    else:
        seed = CFG.seed
    seed_torch(seed)    
   
    train_dataset = SHM_Dataset(CFG["data_path"], mode="train")
    test_dataset = SHM_Dataset(CFG["data_path"], mode="test")
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
    test_dataset = SHM_Dataset(CFG_ck["data_path"], mode="test")
    test_loader = DataLoader(test_dataset, CFG.batch_size,
                             shuffle=False,  num_workers=16, pin_memory=True)
    # MODEL
    model = get_instance(models, 'model', CFG_ck)
    # LOSS
    loss = get_instance(losses, 'loss', CFG_ck)
    
    # TEST
    tester = Trainer(model=model, mode="test", loss=loss, CFG=CFG, checkpoint=checkpoint,
                     test_loader=test_loader, save_path=save_path,show=show)
    tester.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # parser.add_argument('-c', '--config', default='config.yaml', type=str,
    #                     help='Path to the config file (default: config.yaml)')
    # parser.add_argument('-d', '--device', default=None, type=str,
    #                     help='indices of GPUs to enable (default: all)')
    parser.add_argument('-n', '--name', default='imres', type=str,
                        help='the wandb name of run')
    args = parser.parse_args()

    # yaml = YAML(typ='safe')
    with open('/home/tiantong/code/Damage_prediction/config.yaml', encoding='utf-8') as file: 
        CFG = Bunch(safe_load(file))  # 为列表类型
    # os.environ["CUDA_VISIBLE_DEVICES"] = CFG["CUDA_VISIBLE_DEVICES"]
    # if args.device:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.device
     
    wandb.init(project="paper1", config=CFG,
               sync_tensorboard=True, name=f" {CFG['model']['type']} {CFG['loss']['type']} {args.name}")
    path = train(CFG)##   train model
    checkpoint_name = "best_model.pth"##   train model
    # path = "/home/tiantong/code/Damage_prediction/saved/ResNet/230301120633"  ##   test model
    # checkpoint_name = "best_model.pth"  ####   test model
    # test(path, checkpoint_name, CFG,show = True) ###   test model