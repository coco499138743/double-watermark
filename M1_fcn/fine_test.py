import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='dsb2018_96_NestedUNet_woDS',help='model name')
    args = parser.parse_args()
    return args
image_path='inputs/dsb2018_96/images/mytest0.png'
#image_path='inputs/dsb2018_96/w_images/mytest3.png'
img = Image.open(image_path)
transform1 = transforms.Compose([
    #transforms.Resize(96, 96),
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
                                           config['deep_supervision'])

model = model.cuda()
checkpoint=torch.load('models/fine_tune_1/model.pth')
model.load_state_dict(checkpoint)
model.eval()
img = img.cuda()
output = model(img)
image1=Image.open('inputs/dsb2018_96/masks/0/mytest0.png')
#image1=Image.open('inputs/dsb2018_96/w_masks/0/mytest0.png')
#image1=Image.open('mask_yes.png')
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
plt.imshow(image)
image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imwrite("img/outimg1.png",image)
plt.show()

