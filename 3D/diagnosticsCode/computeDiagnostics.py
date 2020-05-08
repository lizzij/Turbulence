# compute diagnostics for samples
import numpy as np
import torch as th
from torch import nn
from tqdm import tqdm
import h5py
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader
from turb_funcs import diagnostics_np
import os, sys


def np_divergence(flow, grid):
    np_Udiv = np.gradient(flow[:, :, :, 0], grid[0])[0]
    np_Vdiv = np.gradient(flow[:, :, :, 0], grid[1])[1]
    np_Wdiv = np.gradient(flow[:, :, :, 0], grid[2])[2]
    np_div = np_Udiv + np_Vdiv + np_Wdiv
    total = np.sum(np_div) / (np.power(128, 3))
    return total


f = h5py.File('/home/arvindm/datasets/ScalarHIT/128cube/scalarHIT_fields100.h5', 'r')
fields = f['fields']
nch = 3
input_dim = (128, 128, 128)
cubesize = 128

dns = fields[5, :, :, :, :3]
mod = fields[50, :, :, :, :3]

# Calculate divergence
dx = (2 * np.pi) / input_dim[0]
dy = (2 * np.pi) / input_dim[0]
dz = (2 * np.pi) / input_dim[0]
dns_div = np_divergence(dns, [dx, dy, dz])
print('DNS divergence is', dns_div)
model_div = np_divergence(mod, [dx, dy, dz])
print('Model divergence is', model_div)

diagnostics_np(mod, dns, save_dir='./diagnostics/', iteration=1, pos=[0, 0, 1], dx=[0.049, 0.049, 0.049],
               diagnostics=['spectrum', 'intermittency', 'QR'])
