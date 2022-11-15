
import argparse
import os
from glob import glob

import cv2
from matplotlib import pyplot as plt
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


def main():
    cudnn.benchmark = True

    # create model
    model = archs.NestedUNet(num_classes=1)
    device = torch.device('cuda:0')
    model.to(device)
    pretrain_dict = torch.load('models/fine_tune/model.pth')
    model.load_state_dict((pretrain_dict), strict=False)
    model.eval()
    # Data loading code
    img_ids = glob(os.path.join('inputs', 'dsb2018_96', 'images', '*' + '.png'))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)



    val_transform = Compose([
        #        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=img_ids,
        img_dir=os.path.join('inputs', 'dsb2018_96', 'images'),
        mask_dir=os.path.join('inputs', 'dsb2018_96', 'masks'),
        img_ext='.png',
        mask_ext='.png',
        num_classes=1,
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        drop_last=False)

    avg_meter = AverageMeter()

    for c in range(1):
        os.makedirs(os.path.join('outputs', 'dsb2018_96_NestedUNet_woDS', str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)

            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(1):
                    cv2.imwrite(os.path.join('outputs', 'dsb2018_96_NestedUNet_woDS', str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % avg_meter.avg)

    plot_examples(input, target, model, num_examples=3)

    torch.cuda.empty_cache()


def plot_examples(datax, datay, model, num_examples=6):  # 画图像
    fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18, 4 * num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        # image_indx=1
        image_arr = model(datax[image_indx:image_indx + 1]).squeeze(0).detach().cpu().numpy()
        ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1, 2, 0))[:, :, 0])
        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(np.squeeze((image_arr > 0.40)[0, :, :].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization")
        ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1, 2, 0))[:, :, 0])
        ax[row_num][2].set_title("Target image")
        image_indx = image_indx + 1
    plt.show()


if __name__ == '__main__':
    main()