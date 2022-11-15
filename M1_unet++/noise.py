import cv2
import numpy as np


def add_gaussian_noise(image_in, noise_sigma):
    """
    给图片添加高斯噪声
    image_in:输入图片
    noise_sigma：
    """
    temp_image = np.float64(np.copy(image_in))

    h, w, _ = temp_image.shape
# 标准正态分布*noise_sigma
    noise = np.random.randn(h, w) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
        noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
        noisy_image[:, :, 2] = temp_image[:, :, 2] + noise

    return noisy_image

img_path = 'inputs/dsb2018_96/images/00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e.png'
img = cv2.imread(img_path)
noise_sigma = 30
noise_img = add_gaussian_noise(img, noise_sigma=noise_sigma)
cv2.imwrite('img1/noise_{}.png'.format(noise_sigma), noise_img)
