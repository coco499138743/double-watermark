import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
image1=Image.open('s.jpg')
image2=Image.open('test02.jpg')
transform2=transforms.Compose([transforms.ToTensor()])
tensor1=transform2(image1)
tensor2=transform2(image2)
print(tensor1)
print(tensor2)
tensor2[2]=tensor1[2]
print(tensor2)
array1=tensor2.numpy()#将tensor数据转为numpy数据
maxValue=array1.max()
array1=array1*255/maxValue#normalize，将图像数据扩展到[0,255]
mat=np.uint8(array1)#float32-->uint8
print('mat_shape:',mat.shape)#mat_shape: (3, 982, 814)
mat=mat.transpose(1,2,0)#mat_shape: (982, 814，3)
mat=cv2.cvtColor(mat,cv2.COLOR_RGB2BGR)
cv2.imshow("img",mat)
cv2.waitKey()