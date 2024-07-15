import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
from skimage.measure import compare_psnr, compare_ssim
import scipy.io as sio






def tensor2img(tensor):
    min_max=(-1, 1)
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = tensor.numpy().transpose(1, 2, 0)
    return tensor


def norm(x):
    x = (x - x.min())/(x.max()-x.min())
    return x * 255.0


def tensor2rgb(tensor):
    min_max=(-1, 1)
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # to range [0,1]
    tensor = tensor.numpy().transpose(1, 2, 0)
    imageR = tensor[:,:,69]
    # imageR = norm(imageR)
    imageG = tensor[:,:,99]
    # imageG = norm(imageG)
    imageB = tensor[:,:,35]
    # imageB = norm(imageB)
    image = cv2.merge([imageR,imageG,imageB])
    image = norm(image)
    return image

def tensor2rgb_band8(tensor):
    min_max=(-1, 1)
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # to range [0,1]
    tensor = tensor.numpy().transpose(1, 2, 0)
    imageR = tensor[:,:,0]
    # imageR = norm(imageR)
    imageG = tensor[:,:,1]
    # imageG = norm(imageG)
    imageB = tensor[:,:,2]
    # imageB = norm(imageB)
    image = cv2.merge([imageR,imageG,imageB])
    image = norm(image)
    return image



def calculate_psnr(x_true, x_pred):
    """
    :param x_true: Input image must have three dimension (H, W, C)
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    channels = x_true.shape[2]
    total_psnr = [compare_psnr(im_test=x_pred[:, :, k],im_true=x_true[:, :, k])
                  for k in range(channels)]

    return np.mean(total_psnr)


def calculate_sam(x_true, x_pred):
    """
    :param x_true: 高光谱图像：格式：(H, W, C)
    :param x_pred: 高光谱图像：格式：(H, W, C)
    :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
    """
    num = 0
    sum_sam = 0
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            tmp_pred = x_pred[x, y].ravel()
            tmp_true = x_true[x, y].ravel()
            if np.linalg.norm(tmp_true) != 0 and np.linalg.norm(tmp_pred) != 0:
                sum_sam += np.arccos(
                    np.inner(tmp_pred, tmp_true) / (np.linalg.norm(tmp_true) * np.linalg.norm(tmp_pred)))
                num += 1
    sam_deg = (sum_sam / num) * 180 / np.pi
    return sam_deg



def calculate_ssim(x_true, x_pred):
    """

    :param x_true:
    :param x_pred:
    :param data_range:
    :param multidimension:
    :return:
    """
    mssim = [compare_ssim(X=x_pred[:, :, i],Y=x_true[:, :, i])
            for i in range(x_true.shape[2])]

    return np.mean(mssim)

def save_img(img, img_path, mode='RGB'):
    # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(img_path, img)
    
def save_mat(mat, mat_path):
    # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    sio.savemat(mat_path,{'SR':mat})


def save_mat_gt(gt, mat, mat_path):
    sio.savemat(mat_path,{'GT':gt, 'SR':mat})
