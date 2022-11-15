from PIL import Image
from torchvision import transforms


image1=Image.open('inputs/dsb2018_96/images/4948e3dfe0483c2198b9616612071ead5474dffd223071cf0a567aea9aca0b9e.png')
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 彩色图像转灰度图像num_output_channels默认1
    transforms.ToTensor()])
image1=transform(image1)
S_M = [1, 58, 1, 5, 2,70, 5, 2, 34, 1, 8, 9]
N=0
for M in S_M:
    print(image1[0][N][M])
    N=N+8
