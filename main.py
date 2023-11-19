from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import os
import os.path
import time
import random
import torch
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
from utils.network import EAPNet1, EAPNet2
from utils.dataset import ReconDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

read_path ='./data/'

dataset_type = 'RS'

'''
Option:
    1. RS : random shape dataset
    2. V: vasculature dataset
    3. ST: sparse targe dataset
    4. AR: acoustic reconstruction dataset
    5. EXP: phantom experiment
'''

if dataset_type == 'RS' or dataset_type == 'AR' or dataset_type == 'EXP':
  circle_mask = torch.tensor(scio.loadmat(read_path + 'Seg_Phantom.mat')['Seg_Phantom'].astype(np.float32)).to(device)
  model = EAPNet1(mask = circle_mask)
else:
  model = EAPNet2()
model = nn.DataParallel(model)
model = model.to(device)

# load network parameters
loadcp=True
if loadcp:
    checkpoint = torch.load(read_path + dataset_type + '/network/model.ckpt')
    model.load_state_dict(checkpoint['state_dict'])


test_dataset = ReconDataset(read_path + dataset_type, dataset_type=dataset_type)
test_batch = 1 # Get one image by one iteration
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch, shuffle=False, drop_last=True)
iterator = iter(test_loader)

print('Number of images:{}'.format(test_dataset.__len__()))

with torch.no_grad():
    (ua_map, p0, mask, num) = next(iterator)  # torch.Size([batch, 1, H, W])

    p0 = p0  # Input images

    outputs = model(p0)

    if dataset_type == 'RS' or dataset_type == 'AR' or dataset_type == 'EXP':
        ua_recon = (outputs.squeeze()) * circle_mask
        vmax = 0.35
    else:
        ua_recon = (outputs.squeeze())
        vmax = 1

    ua_recon = torch.where(torch.isinf(ua_recon), torch.full_like(ua_recon, 0), ua_recon)

    ua_recon = ua_recon.squeeze().cpu() # Reconstructed images
    ua_map = ua_map.squeeze()  # Ground Truth
    p0 = p0.squeeze()

    # Visualization
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(p0, cmap=plt.cm.jet)
    plt.title("p0")

    plt.subplot(1, 3, 2)
    if dataset_type == 'EXP':
        plt.imshow(ua_map, vmin=0, vmax=0.26, cmap=plt.cm.jet)
    else:
        plt.imshow(ua_map, vmin=0, vmax=vmax)
    plt.title("Real ua")

    plt.subplot(1, 3, 3)
    if dataset_type == 'EXP':
        plt.imshow(ua_recon, vmin=0, vmax=0.26, cmap=plt.cm.jet)
    else:
        plt.imshow(ua_recon, vmin=0, vmax=vmax)
    plt.minorticks_on()
    plt.title("Reconstructed ua")
    plt.show()