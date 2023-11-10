import json
import math
import os
import time
from datetime import datetime
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from torch.utils import tensorboard
from tqdm import tqdm
import wandb
from utils.helpers import dir_exists, get_instance, remove_files
from utils.metrics import AverageMeter, get_metrics
import csv



class Trainer:
    def __init__(self, mode, model, resume=None, CFG=None, loss=None,
                 train_loader=None,
                 val_loader=None,
                 checkpoint=None,
                 test_loader=None,
                 save_path=None,show = False):
        self.CFG = CFG

        if self.CFG.amp is True:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss = loss
        self.model = nn.DataParallel(model.cuda())#DataParallel(DP)可以用来实现数据并行方式的分布式训练


        wandb.watch(self.model)
        cudnn.benchmark = True
        # train and val  val是训练过程中的测试集 validation
        if mode == "train":
            # OPTIMIZER
            self.optimizer = get_instance(
                torch.optim, "optimizer", CFG, self.model.parameters())
            self.lr_scheduler = get_instance(
                torch.optim.lr_scheduler, "lr_scheduler", CFG, self.optimizer)


            # MONITORING
            self.improved = True
            self.not_improved_count = 0
            self.mnt_best = -math.inf if self.CFG.mnt_mode == 'max' else math.inf

            # CHECKPOINTS & TENSOBOARD
            start_time = datetime.now().strftime('%y%m%d%H%M%S')
            self.checkpoint_dir = os.path.join(
                CFG.save_dir, self.CFG['model']['type'], start_time)
            self.writer = tensorboard.SummaryWriter(self.checkpoint_dir)
            dir_exists(self.checkpoint_dir)

            # config_save_path = os.path.join(self.checkpoint_dir, 'config.yaml')
            self.train_logger_save_path = os.path.join(
                self.checkpoint_dir, 'runtime.log')
            logger.add(self.train_logger_save_path)
            logger.info(self.checkpoint_dir)
            # with open(config_save_path, 'w') as handle:
            #     json.dump(self.config, handle, indent=4, sort_keys=True)
            if resume:
                self._resume_checkpoint(resume)

        # test
        if mode == "test":
            self.model.load_state_dict(checkpoint['state_dict'])
            self.checkpoint_dir = save_path


    def train(self):
        for epoch in range(1, self.CFG.epochs + 1):
            # RUN TRAIN (AND VAL)
            self._train_epoch(epoch)
            if self.test_loader is not None and epoch % self.CFG.test_per_epochs == 0:
                self.test(epoch=epoch)
            # SAVE CHECKPOINT
            if epoch % self.CFG.save_period == 0:
                self._save_checkpoint(epoch, save_best=self.improved)

        return self.checkpoint_dir

    def _train_epoch(self, epoch):

        self.model.train()
        wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=160)
        for _, data, label in tbar:
            self.data_time.update(time.time() - tic)
            data = data.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)


            # LOSS & OPTIMIZE

            self.optimizer.zero_grad()
            if self.CFG.amp is True:
                with torch.cuda.amp.autocast(enabled=True):  #自动混合精度
                    pre = self.model(data)
                    loss = self.loss(pre, label)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pre = self.model(data)
                loss = self.loss(pre, label)  #label.long()
                loss.backward()
                self.optimizer.step()
            self.total_loss.update(loss.item())
            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()
            # FOR EVAL and INFO
            self._metrics_update(
                *get_metrics(pre, label, threshold=self.CFG.threshold).values())
            # tbar.set_description(
            #     'TRAIN ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
            #         epoch, self.total_loss.average, *
            #         self._metrics_ave().values(), self.batch_time.average,
            #         self.data_time.average))
            # # #下面是多分类指标
            tbar.set_description(
                'TRAIN ({}) | Loss: {:.4f} | Acc {:.4f} F1 {:.4f} Sen {:.4f} Pre {:.4f} Jc {:.4f}|B {:.2f} D {:.2f}  |'.format(
                    epoch, self.total_loss.average, *
                    self._metrics_ave().values(), self.batch_time.average,
                    self.data_time.average))
        self.lr_scheduler.step()
        # METRICS TO TENSORBOARD
        self.writer.add_scalar(
            f'{wrt_mode}/loss', self.total_loss.average, epoch)
        for k, v in list(self._metrics_ave().items())[:-1]:
            self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(
                f'{wrt_mode}/Learning_rate_{i}', opt_group['lr'], epoch)
        # self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)



    def test(self, epoch=1):

        logger.info(f"###### TEST EVALUATION ######")
        wrt_mode = 'test'
        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.test_loader, ncols=160)  #@train 1
        pre_total = []
        lab_total = []
        self._reset_metrics()
        with torch.no_grad():
            for id, data, label in self.test_loader:
                data = data.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                if self.CFG.amp is True:
                    with torch.cuda.amp.autocast(enabled=True):
                        pre = self.model(data)
                        loss = self.loss(pre, label)

                else:
                    pre = self.model(data)
                    loss = self.loss(pre, label)
                              
                self.total_loss.update(loss.item())
                self._metrics_update(
                    *get_metrics(pre, label, threshold=self.CFG.threshold).values())
                # tbar.set_description(
                #     'EVAL ({})  | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f} Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |'.format(
                #         epoch, self.total_loss.average, *self._metrics_ave().values()))
                ##下面是多分类指标
                tbar.set_description( #@train 2
                    'EVAL ({})  | Loss: {:.4f} | Acc {:.4f} F1 {:.4f} Sen {:.4f} Pre {:.4f} Jc {:.4f}  |'.format(
                        epoch, self.total_loss.average, *self._metrics_ave().values()))  #@train2
            
                # _, pre = torch.max(pre.data, 1)   ##test  输出label  pre
                # #save_path='/home/tiantong/data/tt/datanew/testdataout' ##test
                # lab = label.cpu().numpy()##test  输出label  pre
                # lab_total.extend(lab)##test  输出label  pre
                # pre_total.append(pre)##test  输出label  pre
                # for i in range(len(lab)):             ##test  输出label  pre       
                #     if lab[i] != pre[i]:#!=   ==                   ##test  输出label  pre  
                #         print(f"{id[i]}: label {lab[i]},  predit {pre[i]}")##test  输出label  pre

    
        # lab_unique, count = np.unique(lab_total, return_counts=True)
        # for i in range(len(lab_total)):
        #     print (lab_unique[i], count[i])   ##test  输出label  pre      print(lab_unique, count)   
         
        


           
        # LOGGING & TENSORBOARD
        self.total_loss.average
        self.writer.add_scalar(f'{wrt_mode}/loss',  self.total_loss.average, epoch)  ##train  3
        for k, v in list(self._metrics_ave().items())[:-1]:               ##train  3
                self.writer.add_scalar(                                   ##train  3
                    f'{wrt_mode}/{k}', v, epoch)                          ##train  3
        logger.info(f'         loss: {self.total_loss.average}')          ##train  3
        for k, v in self._metrics_ave().items():
            logger.info(f'         {str(k):15s}: {v}')
        log = {
            'test_loss': self.total_loss.average,
            **self._metrics_ave()
        }
        return log

    def _save_checkpoint(self, epoch, save_best=True):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.CFG
        }
        filename = os.path.join(self.checkpoint_dir,
                                'final_checkpoint.pth')
        logger.info(f'Saving a checkpoint: {filename} ...')
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, f'best_model.pth')
            torch.save(state, filename)
            logger.info("Saving current best: best_model.pth")
        return filename

    
    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.auc = AverageMeter()
        self.f1 = AverageMeter()
        self.acc = AverageMeter()
        self.sen = AverageMeter()
        self.jc = AverageMeter()
        self.pre = AverageMeter()
        self.iou = AverageMeter()

    def _metrics_update(self, f1, sen, pre, acc, jc):#, auc, acc, spe, iou
        self.f1.update(f1)
        self.acc.update(acc)
        self.sen.update(sen)        
        self.pre.update(pre)
        # self.auc.update(auc)
        self.jc.update(jc)
        # self.iou.update(iou)

    def _metrics_ave(self):

        return {
            
            "F1": self.f1.average,
            "Acc": self.acc.average,
            "Sen": self.sen.average,           
            "Pre": self.pre.average,
            "jc": self.jc.average,
            # "AUC": self.auc.average,
            # "IOU": self.iou.average
        }
