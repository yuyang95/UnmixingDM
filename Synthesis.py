import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import cv2
import scipy.io as sio

from core.loaddata import HSSampledata
from core.common import *


class decoderAE(nn.Module):
    def __init__(self,input_channels=5, output_channels =128):
        super(decoderAE,self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.decoderlayer = nn.Conv2d(in_channels=self.input_channels,out_channels=self.output_channels,kernel_size=(1,1),bias=False)
	
    def forward(self, x):
        de_result = self.decoderlayer(x)
        return de_result

def norm(x):
    x = (x - x.min())/(x.max()-x.min())
    return x * 255.0

def tensor2rgb(tensor):
    imageR = tensor[:,:,14]
    imageG = tensor[:,:,33]
    imageB = tensor[:,:,53]
    image = cv2.merge([imageR,imageG,imageB])
    image = norm(image)
    return image

# path of synthesized abundance # ./experiments/ddpm/*/mat_results/
train_path    = '' 
# name of checkpoints of the unmixingAE # ./experiments/unmixing/ckpts/*.pth
model_name    = '' 
# save dir of final results
result_path   = './experiments/fusion/HSI/'
image_path    = './experiments/fusion/RGB/'
os.makedirs(result_path, exist_ok=True)
os.makedirs(image_path, exist_ok=True)

sample_set = HSSampledata(image_dir=train_path, augment=False)
sample_loader = DataLoader(sample_set, batch_size=1, num_workers=4, shuffle=False)
ckpt = torch.load(model_name)["model"]
decoderweight = ckpt['decoderlayer.weight']
decoderweight = (decoderweight - decoderweight.min()) / (decoderweight.max() - decoderweight.min())
net = decoderAE()
model_dict = net.state_dict()
model_dict['decoderlayer.weight'] = decoderweight
net.load_state_dict(model_dict)
net.eval().cuda()
device = torch.device('cuda')
print('===> Loading testset')
print('===> Start testing')
with torch.no_grad():
    output = []
    test_number = 0
    # loading model
    for i, (abu) in enumerate(sample_loader):
        abu =  abu.to(device)
        abu = (abu+1)/2
        y = net(abu)
        y = y.squeeze().cpu().numpy().transpose(1, 2, 0)
        filename = str(i).zfill(4)
        abu = abu.squeeze().cpu().numpy().transpose(1, 2, 0)
        
        save_dir = result_path + filename + '.mat'
        rgb_dir = image_path + filename + '.jpg'

        sio.savemat(save_dir,{'HSI':y, 'Abu':abu, 'End':decoderweight})
        cv2.imwrite(rgb_dir,tensor2rgb(y))

def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp
