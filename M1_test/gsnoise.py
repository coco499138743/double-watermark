from skimage import util
import cv2
if __name__ == "__main__":
    img = cv2.imread("img/mytest60.png")
    noise_gs_img1 = util.random_noise(img, mode="gaussian")
    #noise_gs_img = util.random_noise(noise_gs_img1, mode="gaussian")
    #产生高斯噪声，处理后图像变为float64格式
    #cv2.imshow("origin image",img)
    #cv2.imshow("gaussian noise",noise_gs_img )
    noise_gs_img = cv2.normalize(noise_gs_img1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)     #将图像转换为uint8格式，否则保存后是全黑的。
    cv2.imwrite("img/out1.png",noise_gs_img)    #储存图像
    cv2.waitKey(0)
    cv2.destroyAllWindows()