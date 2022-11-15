import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from PIL import Image
from torchvision import transforms
import archs
from metrics import iou_score

from torchvision import transforms

os.environ['KMP_DUPLICATE_LIB_OK']='True'
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='dsb2018_96_UNet_woDS',help='model name')
    args = parser.parse_args()
    return args
#image_path='inputs/dsb2018_96/images/0c2550a23b8a0f29a7575de8c61690d3c31bc897dd5ba66caec201d201a278c2.png'
image_path='inputs/dsb2018_96/images/mytest2.png'
img = Image.open(image_path)
transform1 = transforms.Compose([
#    transforms.Resize(96, 96),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),])
img= transform1(img) #torch.Size([3, 96, 96])
img=img.reshape(1,3,96,96)
args = parse_args()
with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
cudnn.benchmark = True
model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           )

model = model.cuda()
checkpoint=torch.load('models/fine_tune_4/model.pth' )
model.load_state_dict(checkpoint)
model.eval()
img = img.cuda()
output = model(img)
print('output',output.shape)
image1=Image.open('inputs/dsb2018_96/w_masks/0/mytest0.png')
#image1=Image.open('img/out1.png')
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 彩色图像转灰度图像num_output_channels默认1
    transforms.ToTensor()])
image1=transform(image1)
image1=image1.unsqueeze(1)
iou=iou_score(output, image1)
print('iou_score',iou)
print('------1-------->:',output.shape)
print('------2-------->:',image1.shape)
output=output.reshape(1,96,96)
#print('-------2------->:',output.shape)
output = output.data.cpu().numpy()
image = np.array(output).transpose((1, 2, 0))
#print('-------3------->:',output.shape)
mean=np.array([0.5,0.5,0.5])
std=np.array([0.5,0.5,0.5])
image=std*image+mean
image = np.clip(image, 0, 1)
image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)     #将图像转换为uint8格式，否则保存后是全黑的。
cv2.imwrite("img1/mytest.png",image)
plt.imshow(image)
plt.show()
