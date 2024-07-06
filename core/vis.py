import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
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

def save_img(img, img_path, mode='RGB'):
    # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(img_path, img)
    
def save_mat(mat, mat_path):
    # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    sio.savemat(mat_path,{'SR':mat})

from datetime import datetime
def save_mat_gt(gt, mat, mat_path):
    sio.savemat(mat_path,{'GT':gt, 'SR':mat})
