import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
'''tensor3= torch.tensor([[[0.9020, 0.1, 0.1],
                        [0.1,0.1,0.1],
                        [0.1,0.1,0.005]],
                       [[0.8,0.01,0.003],
                        [0.9,0.0,0.0],
                        [0.0,0.0,0.009]],
                       [[0.78,0.1,0.1],
                        [0.1,0.1,0.1],
                        [0.08,0.07,0.24]]])
print(tensor3.shape)
array1=tensor3.numpy()#将tensor数据转为numpy数据
maxValue=array1.max()
array1=array1*255/maxValue#normalize，将图像数据扩展到[0,255]
mat=np.uint8(array1)#float32-->uint8
print('mat_shape:',mat.shape)#mat_shape: (3, 982, 814)
mat=mat.transpose(1,2,0)#mat_shape: (982, 814，3)
mat=cv2.cvtColor(mat,cv2.COLOR_RGB2BGR)
cv2.imshow("img",mat)
cv2.waitKey()'''
image1=Image.open('s.jpg')
transform2=transforms.Compose([transforms.ToTensor()])
tensor2=transform2(image1)
print(tensor2[1])
