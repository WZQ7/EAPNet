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
from utils.dataset import Construct_Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

read_path ='./data/'

dataset_type = 'RS'

'''
Option:
    1. RS : random shape dataset
    2. Va: vasculature dataset
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

train_batch = 32
validation_batch = 16
start_epoch = 0
loadcp = False


train_dataset = Construct_Dataset(dataset_path, dataset_type=dataset_type, data_type='train')
validation_dataset = Construct_Dataset(dataset_path, dataset_type=dataset_type, data_type='validation')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch, shuffle=True, drop_last=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=validation_batch, shuffle=True, drop_last=True)

cudnn.benchmark = True
total_step = len(train_loader)
print("start")
print('train_data :{}'.format(train_dataset.__len__()))
print('validation_data :{}'.format(validation_dataset.__len__()))
end = time.time()

# training implementation
learning_rate = 1*1e-4
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


epoch = start_epoch

Bloss_train = []
Bloss_validation =[]
while epoch < 100:

    for batch_idx, (ua_map, p0, weight) in enumerate(train_loader):
        ua_map = ua_map.to(device)
        p0 = p0.to(device)

        outputs = model(p0)

        # construct weight map of balanced loss
        weight = weight.to(device)
        outputs = outputs / ua_map # normalize intensity
        outputs = outputs * weight # normalize tissue size
        GT = torch.ones(ua_map.shape).to(device) * weight
        Bloss = criterion(outputs, GT)

        # Compute common evaluation metrics
        ae_map = abs(ua_map.detach() - outputs.detach()) # absolute error map
        mae = torch.mean(ae_map.reshape(-1))
        mre = torch.mean((ae_map/(ua_map.detach())).reshape(-1))
        mse = torch.mean(((ua_map.detach() - outputs.detach())**2).reshape(-1))
        msre = torch.mean(((outputs.detach()/ua_map.detach() - 1)**2).reshape(-1))

        # Network parameter upgrade
        optimizer.zero_grad()
        Bloss.backward()
        optimizer.step()

        batch_time=(time.time() - end)
        end = time.time()

        # visualizing intermediate results
        if (batch_idx + 1) % 10 == 0:
          print(f'Epoch [{epoch + 1}], Step [{batch_idx + 1}/{total_step}], Loss: {Bloss.item():.4f},Time:[{batch_time:.4f}]'
                      )

    Bloss_train.append(Bloss.cpu().numpy())

    # perform validation each 10 training epochs
    if (epoch + 1) % 5 == 0:
        print('------------------------------validation--------------------------------')
        with torch.no_grad():
            Bloss_validation_tmp = np.int32(0)
            validation_batch_num = np.int32(0)
            for batch_idx, (ua_map, p0, weight) in enumerate(validation_loader):

                ua_map = ua_map.to(device)
                p0 = p0.to(device)

                outputs = model(p0)

                weight = weight.to(device)
                outputs = outputs / ua_map  # normalize intensity
                outputs = outputs * weight  # normalize tissue size
                GT = torch.ones(ua_map.shape).to(device) * weight
                Bloss = criterion(outputs, GT)

                # Compute common evaluation metrics
                ae_map = abs(ua_map.detach() - outputs.detach()) # absolute error map
                mae = torch.mean(ae_map.reshape(-1))
                mre = torch.mean((ae_map/(ua_map.detach())).reshape(-1))
                mse = torch.mean(((ua_map.detach() - outputs.detach())**2).reshape(-1))
                msre = torch.mean(((outputs.detach()/ua_map.detach() - 1)**2).reshape(-1))

                Bloss_validation_tmp += Bloss.cpu().numpy()
                validation_batch_num += 1

            # output the mean Bloss, not the Bloss of last batch
            Bloss_validation.append(Bloss_validation_tmp/validation_batch_num)

            # visualizing intermediate results
            print(f'Epoch [{epoch + 1}], Loss: {Bloss.item():.4f}')

    # Decay scheme for learning rate
    if (epoch + 1) % 50 == 0:
        learning_rate /= 2
        update_lr(optimizer, learning_rate)

    epoch = epoch+1


