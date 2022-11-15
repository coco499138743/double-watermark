from PIL import Image
from torchvision import transforms


image1=Image.open('img/mytest03.png')
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 彩色图像转灰度图像num_output_channels默认1
    transforms.ToTensor()])
image1=transform(image1)
S_M = [1, 0, 9, 3, 3, 2, 7, 6, 8, 1, 2, 6]
N=0
for M in S_M:
    print(image1[0][N][M])
    N=N+8
