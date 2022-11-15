from PIL import Image
from torchvision import transforms

image1=Image.open('img/mytest60.png')
#image1=Image.open('mask_yes.png')
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 彩色图像转灰度图像num_output_channels默认1
    transforms.ToTensor()])
image1=transform(image1)
image1=image1.unsqueeze(1)
print(image1)
print(image1.shape)
for i in range(96):
    for j in range(96):
        print(image1[0][0][i][j])