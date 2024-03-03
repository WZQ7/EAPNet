from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import os
import os.path
import time
import random
import torch
import torchvision
import torch.nn as nn
from torch.nn import init
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import scipy
import scipy.stats as st
import scipy.io as scio
from collections import OrderedDict
import math
import matplotlib.pyplot as plt
from scipy.io import savemat
from skimage.measure import profile_line


def max_norm(image):
    if image.ndim == 2 or (image.ndim == 3 and image.shape[0] == 1):
        _max = image.max()
        narmal_image = image / _max
    return narmal_image, _max



class Construct_Dataset(data.Dataset):

    def __init__(self, root, dataset_type='RS', data_type='train'):
        self.__ua = []  # Ground Truth
        self.__p0 = []  # p0 images
        self.__tissue = []  # Segmentation images

        self.root = os.path.expanduser(root)
        self.dataset_type = dataset_type

        if data_type == 'train':
            folder = root + "/train_data/"
        elif data_type == 'validation':
            folder = root + "/validation_data/"
        else:
            folder = root + "/examples/"

        for file in os.listdir(folder):
            matdata = scio.loadmat(folder + file)

            p0 = matdata['p0_data']
            ua = matdata['ua_data']
            tissue = matdata['tissue_data']

            __len = p0.shape[0]

            for i in np.arange(__len):
                self.__ua.append(ua[i][np.newaxis, :, :])
                self.__p0.append(p0[i][np.newaxis, :, :])
                self.__tissue.append(tissue[i][np.newaxis, :, :])

    def __getitem__(self, index):
        if self.dataset_type == 'RS' or self.dataset_type == 'AR' or self.dataset_type == 'EXP':

            p0 = self.__p0[index]
            p0, scale = max_norm(p0)
            ua_map = self.__ua[index]
            tissue = self.__tissue[index]

            total_num = (tissue > 0).sum()

            bk = (tissue == 1)
            bk_num = bk.sum()
            tissue_mask = bk
            tissue_num = bk_num
            bk = bk * (total_num / bk_num / 6)

            obj1 = (tissue == 2)  # high contrast
            tissue_mask = np.append(tissue_mask, obj1, axis=0)
            obj1_num = obj1.sum()
            tissue_num = np.hstack((tissue_num, obj1_num))
            obj1 = obj1 * (total_num / obj1_num / 6)

            obj2 = (tissue == 3)  # median contrast
            tissue_mask = np.append(tissue_mask, obj2, axis=0)
            obj2_num = obj2.sum()
            tissue_num = np.hstack((tissue_num, obj2_num))
            obj2 = obj2 * (total_num / obj2_num / 6)

            obj3 = (tissue == 4)  # low contrast
            tissue_mask = np.append(tissue_mask, obj3, axis=0)
            obj3_num = obj3.sum()
            tissue_num = np.hstack((tissue_num, obj3_num))
            obj3 = obj3 * (total_num / obj3_num / 6)

            obj4 = (tissue == 5)  # low contrast
            tissue_mask = np.append(tissue_mask, obj4, axis=0)
            obj4_num = obj4.sum()
            tissue_num = np.hstack((tissue_num, obj4_num))
            obj4 = obj4 * (total_num / obj4_num / 6)

            obj5 = (tissue == 6)  # deep
            tissue_mask = np.append(tissue_mask, obj5, axis=0)
            obj5_num = obj5.sum()
            tissue_num = np.hstack((tissue_num, obj5_num))
            obj5 = obj5 * (total_num / obj5_num / 6)

            weight = bk + obj1 + obj2 + obj3 + obj4 + obj5
        else:
            p0 = self.__p0[index]
            p0, scale = max_norm(p0)
            ua_map = self.__ua[index]
            tissue = self.__tissue

            total_num = 128 * 64

            epi = (tissue[index] == 9)
            tissue_mask = epi
            epi_num = epi.sum(axis=(1, 2))
            tissue_num = epi_num
            epi = epi * (total_num / epi_num / 3)

            derm = np.mod(tissue[index], 2) - (tissue[index] == 9)
            tissue_mask = np.append(tissue_mask, derm, axis=0)
            derm_num = derm.sum(axis=(1, 2))
            tissue_num = np.hstack((tissue_num, derm_num))
            derm = derm * (total_num / derm_num / 3)

            vessel = np.ones(tissue[index].shape) - np.mod(tissue[index], 2)
            tissue_mask = np.append(tissue_mask, vessel, axis=0)
            vessel_num = vessel.sum(axis=(1, 2))
            tissue_num = np.hstack((tissue_num, vessel_num))
            vessel = vessel * (total_num / vessel_num / 3)

            weight = epi + derm + vessel
            weight = np.sqrt(weight)

        ua_map = torch.Tensor(ua_map)
        p0 = torch.Tensor(p0)
        weight = torch.Tensor(weight)

        return ua_map, p0, tissue_mask, tissue_num

    def __len__(self):
        return len(self.__ua)

