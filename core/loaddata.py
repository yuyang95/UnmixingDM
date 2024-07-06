import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os
from core import utils
import cv2

def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])

def datanorm(input):
    output = (input - input.min()) / (input.max() - input.min())
    return output



class HSIDataset(data.Dataset):
    def __init__(self, image_dir, augment=None, use_3D=False):
        self.image_folders = os.listdir(image_dir)        
        self.image_files = []
        for i in self.image_folders:
            if is_mat_file(i):
                full_path = os.path.join(image_dir, i)
                self.image_files.append(full_path)
        self.augment = augment
        self.use_3Dconv = use_3D
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1
    def __getitem__(self, index):
        file_index = index
        aug_num = 0
        if self.augment:
            file_index = index // self.factor
            aug_num = int(index % self.factor)
        load_dir = self.image_files[file_index]
        data = sio.loadmat(load_dir)
        gt = np.array(data['Y'][...], dtype=np.float32)
        gt = datanorm(gt)
        gt = utils.data_augmentation(gt, mode=aug_num)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)

        return gt

    def __len__(self):
        return len(self.image_files)*self.factor


def data_transform(input,min_max=(-1, 1)):
    input = input * (min_max[1] - min_max[0]) + min_max[0]
    return input

class AbuDataset(data.Dataset):
    def __init__(self, image_dir, augment=None, use_3D=False):
        self.image_folders = os.listdir(image_dir)        
        self.image_files = []
        for i in self.image_folders:
            if is_mat_file(i):
                full_path = os.path.join(image_dir, i)
                self.image_files.append(full_path)
        self.augment = augment
        self.use_3Dconv = use_3D
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1
            
        self.length = len(self.image_files)*self.factor
        
    def __getitem__(self, index):
        file_index = index
        aug_num = 0
        if self.augment:
            file_index = index // self.factor
            aug_num = int(index % self.factor)
        load_dir = self.image_files[file_index]
        data = sio.loadmat(load_dir)
        gt = np.array(data['Abu'][...], dtype=np.float32)
        gt = data_transform(gt)

        if self.use_3Dconv:
            gt = gt[np.newaxis, :, :, :]
            gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)
        else:
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)

        return {'Abu': gt}

    def __len__(self):
        return len(self.image_files)*self.factor
    
class HSSampledata(data.Dataset):
    def __init__(self, image_dir, augment=None, use_3D=False):
        self.image_folders = os.listdir(image_dir)        
        self.image_files = []
        for i in self.image_folders:
            if is_mat_file(i):
                full_path = os.path.join(image_dir, i)
                self.image_files.append(full_path)
        self.augment = augment
        self.use_3Dconv = use_3D
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1
    def __getitem__(self, index):
        file_index = index
        aug_num = 0
        if self.augment:
            file_index = index // self.factor
            aug_num = int(index % self.factor)
        load_dir = self.image_files[file_index]
        data = sio.loadmat(load_dir)
        gt = np.array(data['SR'][...], dtype=np.float32)

        if self.use_3Dconv:
            gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)

        else:
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)


        return gt

    def __len__(self):
        return len(self.image_files)*self.factor