import cv2
import math
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import os
from torchvision import transforms

def psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)  # 均方差
    if mse < 1.0e-10:  # 几乎无差异返回100
        return 100
    PIXEL_MAX = 1  # 像素最大值
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def heatmap(img):
    if len(img.shape) == 3:
        b, h, w = img.shape
        heat = np.zeros((b, 3, h, w)).astype('uint8')
        for i in range(b):
            heat[i, :, :, :] = np.transpose(cv2.applyColorMap(img[i, :, :], cv2.COLORMAP_JET), (2, 0, 1))
    else:
        b, c, h, w = img.shape
        heat = np.zeros((b, 3, h, w)).astype('uint8')
        for i in range(b):
            heat[i, :, :, :] = np.transpose(cv2.applyColorMap(img[i, 0, :, :], cv2.COLORMAP_JET), (2, 0, 1))
    return heat


att_path = "./attention"
if not os.path.exists(att_path):
    os.mkdir(att_path)

transform = transforms.Compose([
    # transforms.Resize([256, 512]),
    transforms.ToTensor()])

if __name__ == "__main__":
    MOR_path = "./ckpt/DGNLNet2022-09-15/(DGNLNet2022-09-15) prediction_latest"
    # MOR_path = "./dataset/test/A"
    gt_path = "./MOR_new/test/gt"
    images_name = os.listdir(MOR_path)
    psnr_sum = 0
    ssim_sum = 0
    i = 0
    # 遍历所有文件名
    for eachname in images_name:
        # 按照规则将内容写入txt文件中

        filename = eachname.split('-')[1]
        trainname = MOR_path + "/" + filename
        filename2 = filename.split('_rain_')[0] + '.png'
        gtname = gt_path + "/" + filename2

        # trainname = MOR_path + "/" + strlist[0] + "-" + strlist[1]
        # gtname = gt_path + "/" + strlist[0] + "-" + strlist[1]

        img1 = cv2.imread(trainname)
        img2 = cv2.imread(gtname)

        mask = (img1 - img2).sum(axis=2)
        mask[mask <= 30] = 0
        mask[mask > 30] = 1
        mask = mask.astype(np.float32)
        h, w = mask.shape
        mask = transform(mask).unsqueeze(0).cuda()
        attention = mask.data.squeeze(0).cpu()
        attention = attention.cpu().data
        attention = attention * 255
        attention = np.clip(attention.numpy(), 0, 255).astype('uint8')
        attention = attention.reshape(1, h, w)
        attention = heatmap(attention)

        attention = np.transpose(attention[0], (1, 2, 0))

        # cv2.imshow('image',attention)

        # cv2.waitKey(0)
        cv2.imwrite(att_path+filename, attention)

        psnrnum = compare_psnr(img1, img2, data_range=255)
        ssimnum = compare_ssim(img1, img2, data_range=255, multichannel=True)
        psnr_sum = psnr_sum + psnrnum
        ssim_sum = ssim_sum + ssimnum
        i = i + 1
        print(i)
        print(trainname + ":", psnrnum)
        print(gtname + ":", ssimnum)
        if (i == len(images_name) / 2):
            break
    finalpsnr = psnr_sum / (len(images_name) / 2)
    fianlssim = ssim_sum / (len(images_name) / 2)
    print(finalpsnr)
    print(fianlssim)
