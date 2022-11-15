import argparse
import os
from glob import glob

import pandas as pd
from torch import optim

import losses
from dataset import Dataset
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from metrics import iou_score
from utils import AverageMeter
import torch
import torch.backends.cudnn as cudnn
import archs
from collections import OrderedDict
from skimage.metrics import structural_similarity as ssim
import numpy as np
import random

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf

model= archs.UNet(num_classes=1)
device = torch.device('cuda:0')
model.to(device)
#model_dict=model.state_dict()
pretrain_dict=torch.load('models/dsb2018_96_UNet_woDS/model.pth')
model.load_state_dict((pretrain_dict),strict=False)

'''model_dict['conv0_0.conv1.weight']=pretrain_dict['conv0_0.conv1.weight']
model_dict['conv0_0.conv1.bias']=pretrain_dict['conv0_0.conv1.bias']
model_dict['conv0_0.bn1.weight']=pretrain_dict['conv0_0.bn1.weight']
model_dict['conv0_0.bn1.bias']=pretrain_dict['conv0_0.bn1.bias']
model_dict['conv0_0.bn1.running_mean']=pretrain_dict['conv0_0.bn1.running_mean']
model_dict['conv0_0.bn1.running_var']=pretrain_dict['conv0_0.bn1.running_var']
model_dict['conv0_0.bn1.num_batches_tracked']=pretrain_dict['conv0_0.bn1.num_batches_tracked']
model_dict['conv0_0.conv2.weight']=pretrain_dict['conv0_0.conv2.weight']
model_dict['conv0_0.conv2.bias']=pretrain_dict['conv0_0.conv2.bias']
model_dict['conv0_0.bn2.weight']=pretrain_dict['conv0_0.bn2.weight']
model_dict['conv0_0.bn2.bias']=pretrain_dict['conv0_0.bn2.bias']
model_dict['conv0_0.bn2.running_mean']=pretrain_dict['conv0_0.bn2.running_mean']
model_dict['conv0_0.bn2.running_var']=pretrain_dict['conv0_0.bn2.running_var']
model_dict['conv0_0.bn2.num_batches_tracked']=pretrain_dict['conv0_0.bn2.num_batches_tracked']

model_dict['conv1_0.conv1.weight']=pretrain_dict['conv1_0.conv1.weight']
model_dict['conv1_0.conv1.bias']=pretrain_dict['conv1_0.conv1.bias']
model_dict['conv1_0.bn1.weight']=pretrain_dict['conv1_0.bn1.weight']
model_dict['conv1_0.bn1.bias']=pretrain_dict['conv1_0.bn1.bias']
model_dict['conv1_0.bn1.running_mean']=pretrain_dict['conv1_0.bn1.running_mean']
model_dict['conv1_0.bn1.running_var']=pretrain_dict['conv1_0.bn1.running_var']
model_dict['conv1_0.bn1.num_batches_tracked']=pretrain_dict['conv1_0.bn1.num_batches_tracked']
model_dict['conv1_0.conv2.weight']=pretrain_dict['conv1_0.conv2.weight']
model_dict['conv1_0.conv2.bias']=pretrain_dict['conv1_0.conv2.bias']
model_dict['conv1_0.bn2.weight']=pretrain_dict['conv1_0.bn2.weight']
model_dict['conv1_0.bn2.bias']=pretrain_dict['conv1_0.bn2.bias']
model_dict['conv1_0.bn2.running_mean']=pretrain_dict['conv1_0.bn2.running_mean']
model_dict['conv1_0.bn2.running_var']=pretrain_dict['conv1_0.bn2.running_var']
model_dict['conv1_0.bn2.num_batches_tracked']=pretrain_dict['conv1_0.bn2.num_batches_tracked']

model_dict['conv2_0.conv1.weight']=pretrain_dict['conv2_0.conv1.weight']
model_dict['conv2_0.conv1.bias']=pretrain_dict['conv2_0.conv1.bias']
model_dict['conv2_0.bn1.weight']=pretrain_dict['conv2_0.bn1.weight']
model_dict['conv2_0.bn1.bias']=pretrain_dict['conv2_0.bn1.bias']
model_dict['conv2_0.bn1.running_mean']=pretrain_dict['conv2_0.bn1.running_mean']
model_dict['conv2_0.bn1.running_var']=pretrain_dict['conv2_0.bn1.running_var']
model_dict['conv2_0.bn1.num_batches_tracked']=pretrain_dict['conv2_0.bn1.num_batches_tracked']
model_dict['conv2_0.conv2.weight']=pretrain_dict['conv2_0.conv2.weight']
model_dict['conv2_0.conv2.bias']=pretrain_dict['conv2_0.conv2.bias']
model_dict['conv2_0.bn2.weight']=pretrain_dict['conv2_0.bn2.weight']
model_dict['conv2_0.bn2.bias']=pretrain_dict['conv2_0.bn2.bias']
model_dict['conv2_0.bn2.running_mean']=pretrain_dict['conv2_0.bn2.running_mean']
model_dict['conv2_0.bn2.running_var']=pretrain_dict['conv2_0.bn2.running_var']
model_dict['conv2_0.bn2.num_batches_tracked']=pretrain_dict['conv2_0.bn2.num_batches_tracked']

model_dict['conv3_0.conv1.weight']=pretrain_dict['conv3_0.conv1.weight']
model_dict['conv3_0.conv1.bias']=pretrain_dict['conv3_0.conv1.bias']
model_dict['conv3_0.bn1.weight']=pretrain_dict['conv3_0.bn1.weight']
model_dict['conv3_0.bn1.bias']=pretrain_dict['conv3_0.bn1.bias']
model_dict['conv3_0.bn1.running_mean']=pretrain_dict['conv3_0.bn1.running_mean']
model_dict['conv3_0.bn1.running_var']=pretrain_dict['conv3_0.bn1.running_var']
model_dict['conv3_0.bn1.num_batches_tracked']=pretrain_dict['conv3_0.bn1.num_batches_tracked']
model_dict['conv3_0.conv2.weight']=pretrain_dict['conv3_0.conv2.weight']
model_dict['conv3_0.conv2.bias']=pretrain_dict['conv3_0.conv2.bias']
model_dict['conv3_0.bn2.weight']=pretrain_dict['conv3_0.bn2.weight']
model_dict['conv3_0.bn2.bias']=pretrain_dict['conv3_0.bn2.bias']
model_dict['conv3_0.bn2.running_mean']=pretrain_dict['conv3_0.bn2.running_mean']
model_dict['conv3_0.bn2.running_var']=pretrain_dict['conv3_0.bn2.running_var']
model_dict['conv3_0.bn2.num_batches_tracked']=pretrain_dict['conv3_0.bn2.num_batches_tracked']

model_dict['conv4_0.conv1.weight']=pretrain_dict['conv4_0.conv1.weight']
model_dict['conv4_0.conv1.bias']=pretrain_dict['conv4_0.conv1.bias']
model_dict['conv4_0.bn1.weight']=pretrain_dict['conv4_0.bn1.weight']
model_dict['conv4_0.bn1.bias']=pretrain_dict['conv4_0.bn1.bias']
model_dict['conv4_0.bn1.running_mean']=pretrain_dict['conv4_0.bn1.running_mean']
model_dict['conv4_0.bn1.running_var']=pretrain_dict['conv4_0.bn1.running_var']
model_dict['conv4_0.bn1.num_batches_tracked']=pretrain_dict['conv4_0.bn1.num_batches_tracked']
model_dict['conv4_0.conv2.weight']=pretrain_dict['conv4_0.conv2.weight']
model_dict['conv4_0.conv2.bias']=pretrain_dict['conv4_0.conv2.bias']
model_dict['conv4_0.bn2.weight']=pretrain_dict['conv4_0.bn2.weight']
model_dict['conv4_0.bn2.bias']=pretrain_dict['conv4_0.bn2.bias']
model_dict['conv4_0.bn2.running_mean']=pretrain_dict['conv4_0.bn2.running_mean']
model_dict['conv4_0.bn2.running_var']=pretrain_dict['conv4_0.bn2.running_var']
model_dict['conv4_0.bn2.num_batches_tracked']=pretrain_dict['conv4_0.bn2.num_batches_tracked']

model_dict['conv0_1.conv1.weight']=pretrain_dict['conv0_1.conv1.weight']
model_dict['conv0_1.conv1.bias']=pretrain_dict['conv0_1.conv1.bias']
model_dict['conv0_1.bn1.weight']=pretrain_dict['conv0_1.bn1.weight']
model_dict['conv0_1.bn1.bias']=pretrain_dict['conv0_1.bn1.bias']
model_dict['conv0_1.bn1.running_mean']=pretrain_dict['conv0_1.bn1.running_mean']
model_dict['conv0_1.bn1.running_var']=pretrain_dict['conv0_1.bn1.running_var']
model_dict['conv0_1.bn1.num_batches_tracked']=pretrain_dict['conv0_1.bn1.num_batches_tracked']
model_dict['conv0_1.conv2.weight']=pretrain_dict['conv0_1.conv2.weight']
model_dict['conv0_1.conv2.bias']=pretrain_dict['conv0_1.conv2.bias']
model_dict['conv0_1.bn2.weight']=pretrain_dict['conv0_1.bn2.weight']
model_dict['conv0_1.bn2.bias']=pretrain_dict['conv0_1.bn2.bias']
model_dict['conv0_1.bn2.running_mean']=pretrain_dict['conv0_1.bn2.running_mean']
model_dict['conv0_1.bn2.running_var']=pretrain_dict['conv0_1.bn2.running_var']
model_dict['conv0_1.bn2.num_batches_tracked']=pretrain_dict['conv0_1.bn2.num_batches_tracked']

model_dict['conv1_1.conv1.weight']=pretrain_dict['conv1_1.conv1.weight']
model_dict['conv1_1.conv1.bias']=pretrain_dict['conv1_1.conv1.bias']
model_dict['conv1_1.bn1.weight']=pretrain_dict['conv1_1.bn1.weight']
model_dict['conv1_1.bn1.bias']=pretrain_dict['conv1_1.bn1.bias']
model_dict['conv1_1.bn1.running_mean']=pretrain_dict['conv1_1.bn1.running_mean']
model_dict['conv1_1.bn1.running_var']=pretrain_dict['conv1_1.bn1.running_var']
model_dict['conv1_1.bn1.num_batches_tracked']=pretrain_dict['conv1_1.bn1.num_batches_tracked']
model_dict['conv1_1.conv2.weight']=pretrain_dict['conv1_1.conv2.weight']
model_dict['conv1_1.conv2.bias']=pretrain_dict['conv1_1.conv2.bias']
model_dict['conv1_1.bn2.weight']=pretrain_dict['conv1_1.bn2.weight']
model_dict['conv1_1.bn2.bias']=pretrain_dict['conv1_1.bn2.bias']
model_dict['conv1_1.bn2.running_mean']=pretrain_dict['conv1_1.bn2.running_mean']
model_dict['conv1_1.bn2.running_var']=pretrain_dict['conv1_1.bn2.running_var']
model_dict['conv1_1.bn2.num_batches_tracked']=pretrain_dict['conv1_1.bn2.num_batches_tracked']

model_dict['conv2_1.conv1.weight']=pretrain_dict['conv2_1.conv1.weight']
model_dict['conv2_1.conv1.bias']=pretrain_dict['conv2_1.conv1.bias']
model_dict['conv2_1.bn1.weight']=pretrain_dict['conv2_1.bn1.weight']
model_dict['conv2_1.bn1.bias']=pretrain_dict['conv2_1.bn1.bias']
model_dict['conv2_1.bn1.running_mean']=pretrain_dict['conv2_1.bn1.running_mean']
model_dict['conv2_1.bn1.running_var']=pretrain_dict['conv2_1.bn1.running_var']
model_dict['conv2_1.bn1.num_batches_tracked']=pretrain_dict['conv2_1.bn1.num_batches_tracked']
model_dict['conv2_1.conv2.weight']=pretrain_dict['conv2_1.conv2.weight']
model_dict['conv2_1.conv2.bias']=pretrain_dict['conv2_1.conv2.bias']
model_dict['conv2_1.bn2.weight']=pretrain_dict['conv2_1.bn2.weight']
model_dict['conv2_1.bn2.bias']=pretrain_dict['conv2_1.bn2.bias']
model_dict['conv2_1.bn2.running_mean']=pretrain_dict['conv2_1.bn2.running_mean']
model_dict['conv2_1.bn2.running_var']=pretrain_dict['conv2_1.bn2.running_var']
model_dict['conv2_1.bn2.num_batches_tracked']=pretrain_dict['conv2_1.bn2.num_batches_tracked']

model_dict['conv3_1.conv1.weight']=pretrain_dict['conv3_1.conv1.weight']
model_dict['conv3_1.conv1.bias']=pretrain_dict['conv3_1.conv1.bias']
model_dict['conv3_1.bn1.weight']=pretrain_dict['conv3_1.bn1.weight']
model_dict['conv3_1.bn1.bias']=pretrain_dict['conv3_1.bn1.bias']
model_dict['conv3_1.bn1.running_mean']=pretrain_dict['conv3_1.bn1.running_mean']
model_dict['conv3_1.bn1.running_var']=pretrain_dict['conv3_1.bn1.running_var']
model_dict['conv3_1.bn1.num_batches_tracked']=pretrain_dict['conv3_1.bn1.num_batches_tracked']
model_dict['conv3_1.conv2.weight']=pretrain_dict['conv3_1.conv2.weight']
model_dict['conv3_1.conv2.bias']=pretrain_dict['conv3_1.conv2.bias']
model_dict['conv3_1.bn2.weight']=pretrain_dict['conv3_1.bn2.weight']
model_dict['conv3_1.bn2.bias']=pretrain_dict['conv3_1.bn2.bias']
model_dict['conv3_1.bn2.running_mean']=pretrain_dict['conv3_1.bn2.running_mean']
model_dict['conv3_1.bn2.running_var']=pretrain_dict['conv3_1.bn2.running_var']
model_dict['conv3_1.bn2.num_batches_tracked']=pretrain_dict['conv3_1.bn2.num_batches_tracked']

model_dict['conv0_2.conv1.weight']=pretrain_dict['conv0_2.conv1.weight']
model_dict['conv0_2.conv1.bias']=pretrain_dict['conv0_2.conv1.bias']
model_dict['conv0_2.bn1.weight']=pretrain_dict['conv0_2.bn1.weight']
model_dict['conv0_2.bn1.bias']=pretrain_dict['conv0_2.bn1.bias']
model_dict['conv0_2.bn1.running_mean']=pretrain_dict['conv0_2.bn1.running_mean']
model_dict['conv0_2.bn1.running_var']=pretrain_dict['conv0_2.bn1.running_var']
model_dict['conv0_2.bn1.num_batches_tracked']=pretrain_dict['conv0_2.bn1.num_batches_tracked']
model_dict['conv0_2.conv2.weight']=pretrain_dict['conv0_2.conv2.weight']
model_dict['conv0_2.conv2.bias']=pretrain_dict['conv0_2.conv2.bias']
model_dict['conv0_2.bn2.weight']=pretrain_dict['conv0_2.bn2.weight']
model_dict['conv0_2.bn2.bias']=pretrain_dict['conv0_2.bn2.bias']
model_dict['conv0_2.bn2.running_mean']=pretrain_dict['conv0_2.bn2.running_mean']
model_dict['conv0_2.bn2.running_var']=pretrain_dict['conv0_2.bn2.running_var']
model_dict['conv0_2.bn2.num_batches_tracked']=pretrain_dict['conv0_2.bn2.num_batches_tracked']

model_dict['conv1_2.conv1.weight']=pretrain_dict['conv1_2.conv1.weight']
model_dict['conv1_2.conv1.bias']=pretrain_dict['conv1_2.conv1.bias']
model_dict['conv1_2.bn1.weight']=pretrain_dict['conv1_2.bn1.weight']
model_dict['conv1_2.bn1.bias']=pretrain_dict['conv1_2.bn1.bias']
model_dict['conv1_2.bn1.running_mean']=pretrain_dict['conv1_2.bn1.running_mean']
model_dict['conv1_2.bn1.running_var']=pretrain_dict['conv1_2.bn1.running_var']
model_dict['conv1_2.bn1.num_batches_tracked']=pretrain_dict['conv1_2.bn1.num_batches_tracked']
model_dict['conv1_2.conv2.weight']=pretrain_dict['conv1_2.conv2.weight']
model_dict['conv1_2.conv2.bias']=pretrain_dict['conv1_2.conv2.bias']
model_dict['conv1_2.bn2.weight']=pretrain_dict['conv1_2.bn2.weight']
model_dict['conv1_2.bn2.bias']=pretrain_dict['conv1_2.bn2.bias']
model_dict['conv1_2.bn2.running_mean']=pretrain_dict['conv1_2.bn2.running_mean']
model_dict['conv1_2.bn2.running_var']=pretrain_dict['conv1_2.bn2.running_var']
model_dict['conv1_2.bn2.num_batches_tracked']=pretrain_dict['conv1_2.bn2.num_batches_tracked']

model_dict['conv2_2.conv1.weight']=pretrain_dict['conv2_2.conv1.weight']
model_dict['conv2_2.conv1.bias']=pretrain_dict['conv2_2.conv1.bias']
model_dict['conv2_2.bn1.weight']=pretrain_dict['conv2_2.bn1.weight']
model_dict['conv2_2.bn1.bias']=pretrain_dict['conv2_2.bn1.bias']
model_dict['conv2_2.bn1.running_mean']=pretrain_dict['conv2_2.bn1.running_mean']
model_dict['conv2_2.bn1.running_var']=pretrain_dict['conv2_2.bn1.running_var']
model_dict['conv2_2.bn1.num_batches_tracked']=pretrain_dict['conv2_2.bn1.num_batches_tracked']
model_dict['conv2_2.conv2.weight']=pretrain_dict['conv2_2.conv2.weight']
model_dict['conv2_2.conv2.bias']=pretrain_dict['conv2_2.conv2.bias']
model_dict['conv2_2.bn2.weight']=pretrain_dict['conv2_2.bn2.weight']
model_dict['conv2_2.bn2.bias']=pretrain_dict['conv2_2.bn2.bias']
model_dict['conv2_2.bn2.running_mean']=pretrain_dict['conv2_2.bn2.running_mean']
model_dict['conv2_2.bn2.running_var']=pretrain_dict['conv2_2.bn2.running_var']
model_dict['conv2_2.bn2.num_batches_tracked']=pretrain_dict['conv2_2.bn2.num_batches_tracked']

model_dict['conv0_3.conv1.weight']=pretrain_dict['conv0_3.conv1.weight']
model_dict['conv0_3.conv1.bias']=pretrain_dict['conv0_3.conv1.bias']
model_dict['conv0_3.bn1.weight']=pretrain_dict['conv0_3.bn1.weight']
model_dict['conv0_3.bn1.bias']=pretrain_dict['conv0_3.bn1.bias']
model_dict['conv0_3.bn1.running_mean']=pretrain_dict['conv0_3.bn1.running_mean']
model_dict['conv0_3.bn1.running_var']=pretrain_dict['conv0_3.bn1.running_var']
model_dict['conv0_3.bn1.num_batches_tracked']=pretrain_dict['conv0_3.bn1.num_batches_tracked']
model_dict['conv0_3.conv2.weight']=pretrain_dict['conv0_3.conv2.weight']
model_dict['conv0_3.conv2.bias']=pretrain_dict['conv0_3.conv2.bias']
model_dict['conv0_3.bn2.weight']=pretrain_dict['conv0_3.bn2.weight']
model_dict['conv0_3.bn2.bias']=pretrain_dict['conv0_3.bn2.bias']
model_dict['conv0_3.bn2.running_mean']=pretrain_dict['conv0_3.bn2.running_mean']
model_dict['conv0_3.bn2.running_var']=pretrain_dict['conv0_3.bn2.running_var']
model_dict['conv0_3.bn2.num_batches_tracked']=pretrain_dict['conv0_3.bn2.num_batches_tracked']

model_dict['conv1_3.conv1.weight']=pretrain_dict['conv1_3.conv1.weight']
model_dict['conv1_3.conv1.bias']=pretrain_dict['conv1_3.conv1.bias']
model_dict['conv1_3.bn1.weight']=pretrain_dict['conv1_3.bn1.weight']
model_dict['conv1_3.bn1.bias']=pretrain_dict['conv1_3.bn1.bias']
model_dict['conv1_3.bn1.running_mean']=pretrain_dict['conv1_3.bn1.running_mean']
model_dict['conv1_3.bn1.running_var']=pretrain_dict['conv1_3.bn1.running_var']
model_dict['conv1_3.bn1.num_batches_tracked']=pretrain_dict['conv1_3.bn1.num_batches_tracked']
model_dict['conv1_3.conv2.weight']=pretrain_dict['conv1_3.conv2.weight']
model_dict['conv1_3.conv2.bias']=pretrain_dict['conv1_3.conv2.bias']
model_dict['conv1_3.bn2.weight']=pretrain_dict['conv1_3.bn2.weight']
model_dict['conv1_3.bn2.bias']=pretrain_dict['conv1_3.bn2.bias']
model_dict['conv1_3.bn2.running_mean']=pretrain_dict['conv1_3.bn2.running_mean']
model_dict['conv1_3.bn2.running_var']=pretrain_dict['conv1_3.bn2.running_var']
model_dict['conv1_3.bn2.num_batches_tracked']=pretrain_dict['conv1_3.bn2.num_batches_tracked']

model_dict['conv0_4.conv1.weight']=pretrain_dict['conv0_4.conv1.weight']
model_dict['conv0_4.conv1.bias']=pretrain_dict['conv0_4.conv1.bias']
model_dict['conv0_4.bn1.weight']=pretrain_dict['conv0_4.bn1.weight']
model_dict['conv0_4.bn1.bias']=pretrain_dict['conv0_4.bn1.bias']
model_dict['conv0_4.bn1.running_mean']=pretrain_dict['conv0_4.bn1.running_mean']
model_dict['conv0_4.bn1.running_var']=pretrain_dict['conv0_4.bn1.running_var']
model_dict['conv0_4.bn1.num_batches_tracked']=pretrain_dict['conv0_4.bn1.num_batches_tracked']
model_dict['conv0_4.conv2.weight']=pretrain_dict['conv0_4.conv2.weight']
model_dict['conv0_4.conv2.bias']=pretrain_dict['conv0_4.conv2.bias']
model_dict['conv0_4.bn2.weight']=pretrain_dict['conv0_4.bn2.weight']
model_dict['conv0_4.bn2.bias']=pretrain_dict['conv0_4.bn2.bias']
model_dict['conv0_4.bn2.running_mean']=pretrain_dict['conv0_4.bn2.running_mean']
model_dict['conv0_4.bn2.running_var']=pretrain_dict['conv0_4.bn2.running_var']
model_dict['conv0_4.bn2.num_batches_tracked']=pretrain_dict['conv0_4.bn2.num_batches_tracked']

model_dict['final.weight']=pretrain_dict['final.weight']
model_dict['final.bias']=pretrain_dict['final.bias']
'''


for name,param in model.named_parameters():
    if "conv2_0"  in name:
        param.requies_grad = True
    else :
        param.requies_grad =False
for name,param in model.named_parameters():
    if param.requies_grad:
        print("requires_grade:T",name)
    else :
        print("requires_grade:F",name)

optimizer =optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=1e-3)

def train(train_loader,watermark_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss1 = criterion(output, target)
        iou = iou_score(output, target)
        for input2, target2, _ in watermark_loader:
            input2 = input2.cuda()
            target2 = target2.cuda()
            output2 = model(input2)
            loss2 = 1-iou_score(output2, target2)
        f_lambda1 = 1
        f_lambda2=0.4
        loss = f_lambda1 * loss1 +f_lambda2*loss2
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)

    pbar.close()
    torch.save(model.state_dict(), 'models/fine_tune_1/model.pth')
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])
def validate(val_loader,watermark_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(val_loader))
    for input, target, _ in val_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss1 = criterion(output, target)
        iou = iou_score(output, target)
        for input2, target2, _ in watermark_loader:
            input2 = input2.cuda()
            target2 = target2.cuda()
            output2 = model(input2)
            loss2 = 1-iou_score(output2, target2)
        f_lambda1 = 1
        f_lambda2=0.4
        loss = f_lambda1 * loss1 +f_lambda2*loss2
        # compute gradient and do optimizing step
        loss.backward()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)

    pbar.close()
    torch.save(model.state_dict(), 'models/fine_tune_1/model.pth')
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])
img_ids = glob(os.path.join('inputs', 'dsb2018_96', 'images', '*' + '.png'))
img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
train_transform = Compose([
        #transforms.RandomRotate90(),
        transforms.Flip(),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
        ], p=1),#按照归一化的概率选择执行哪一个
        transforms.Normalize(),
    ])
val_transform = Compose([
        transforms.Normalize(),
    ])
train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', 'dsb2018_96', 'images'),
        mask_dir=os.path.join('inputs', 'dsb2018_96', 'masks'),
        img_ext='.png',
        mask_ext='.png',
        num_classes=1,
        transform=train_transform
        )
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        drop_last=True)
val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', 'dsb2018_96', 'images'),
        mask_dir=os.path.join('inputs', 'dsb2018_96', 'masks'),
        img_ext='.png',
        mask_ext='.png',
        num_classes=1,
        transform=val_transform)
val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        drop_last=False)
watermark_ids = glob(os.path.join('inputs', 'dsb2018_96', 'w_images', '*' + '.png'))
watermark_ids = [os.path.splitext(os.path.basename(p))[0] for p in watermark_ids]
watermark_dataset = Dataset(
    img_ids=watermark_ids,
    img_dir=os.path.join('inputs', 'dsb2018_96', 'w_images'),
    mask_dir=os.path.join('inputs', 'dsb2018_96', 'w_masks'),
    img_ext='.png',
    mask_ext='.png',
    num_classes=1,
)
watermark_loader = torch.utils.data.DataLoader(
        watermark_dataset,
        batch_size=20,
        shuffle=True,
        num_workers=0,
        drop_last=True)
criterion=losses.BCEDiceLoss().cuda()

log = OrderedDict([
    ('epoch', []),
    ('loss', []),
    ('iou', []),
    ('val_loss', []),
    ('val_iou', []),


])

for epoch in range(200):
    print('Epoch [%d/200]' % (epoch))
    train_log = train(train_loader,watermark_loader, model, criterion, optimizer)
    val_log = validate(val_loader, watermark_loader, model, criterion)
    torch.cuda.empty_cache()
    log['epoch'].append(epoch)
    log['loss'].append(train_log['loss'])
    log['iou'].append(train_log['iou'])
    log['val_loss'].append(val_log['loss'])
    log['val_iou'].append(val_log['iou'])


    pd.DataFrame(log).to_csv('models/fine_tune_1/log.csv'
                             , index=False)
