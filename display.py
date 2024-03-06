
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from utils.network import EAPNet
from utils.dataset import Construct_Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
read_path = './data/'
dataset_type = 'RS'
example_index = 1  # choose a certain example image of the selected dataset, up to 3
'''
Option:
    1. RS : random shape dataset, including 2 example images
    2. Va: vasculature dataset, including 3 example images
    3. ST: sparse targe dataset, including 3 example images
    4. AR: acoustic reconstruction dataset, including 2 example images
    5. EXP: phantom experiment, including 2 example images
    6. mouse: in vivo experiment, including 2 example images
'''

model = EAPNet()
model = nn.DataParallel(model)
model = model.to(device)

# load network parameters
loadcp = True
if loadcp:
    checkpoint = torch.load(read_path + dataset_type + '/network/model.ckpt')
    model.load_state_dict(checkpoint['state_dict'])

if dataset_type == 'RS' or dataset_type == 'AR' or dataset_type == 'EXP':
    obj_seg = torch.tensor(scio.loadmat(read_path + 'Seg_Phantom.mat')['Seg_Phantom'].astype(np.float32))
elif dataset_type == 'mouse':
    obj_seg = torch.tensor(scio.loadmat(read_path + 'Seg_mouse.mat')['Seg_mouse'].astype(np.float32))
    obj_seg = obj_seg[example_index-1]
else:
    obj_seg = None

test_dataset = Construct_Dataset(read_path + dataset_type, dataset_type=dataset_type, data_type='test')
test_batch = 1  # Get one image by one iteration
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch, shuffle=False, drop_last=True)
iterator = iter(test_loader)

print('Number of images:{}'.format(test_dataset.__len__()))

with torch.no_grad():
    for i in range(example_index):
        (ua_map, p0, mask, num) = next(iterator)  # torch.Size([batch, 1, H, W])
    if dataset_type == 'mouse':
        p0 = p0*obj_seg
    start_time = time.time()
    outputs = model(p0, obj_seg)
    estimate_time = time.time() - start_time

    if dataset_type == 'RS' or dataset_type == 'AR' or dataset_type == 'EXP':
        ua_recon = (outputs.squeeze()) * obj_seg.to(device)
        vmax = 0.35
    else:
        ua_recon = (outputs.squeeze())
        vmax = 1

    ua_recon = torch.where(torch.isinf(ua_recon), torch.full_like(ua_recon, 0), ua_recon)
    ua_recon = ua_recon.squeeze().cpu() # Reconstructed images
    p0 = p0.squeeze()
    if dataset_type != 'mouse':
        ua_map = ua_map.squeeze()  # Ground Truth

    # Visualization
    plt.figure()
    plt.subplot(1, 3, 1)
    if dataset_type != 'mouse':
        plt.imshow(p0, vmin=0, cmap=plt.cm.jet)
    else:
        plt.imshow(p0, vmin=0, cmap=plt.cm.gray)
    plt.title("p0")

    if dataset_type != 'mouse':
        plt.subplot(1, 3, 2)
        if dataset_type == 'EXP':
            plt.imshow(ua_map, vmin=0, vmax=0.26, cmap=plt.cm.jet)
        else:
            plt.imshow(ua_map, vmin=0, vmax=vmax)
        plt.title("Real ua")

    plt.subplot(1, 3, 3)
    if dataset_type == 'EXP':
        plt.imshow(ua_recon, vmin=0, vmax=0.26, cmap=plt.cm.jet)
    elif dataset_type == 'mouse':
        plt.imshow(ua_recon, vmin=0, cmap=plt.cm.gray)
    else:
        plt.imshow(ua_recon, vmin=0, vmax=vmax)
    plt.minorticks_on()
    plt.title("Reconstructed ua")
    plt.show()

    print(f'time of estimate one image: {estimate_time}')