import numpy as np
import pandas as pd
#!{sys.executable} -m pip install --user netCDF
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import torch
import matplotlib.animation as animation
import torch.nn.functional as F
import random
import os
import imageio
from tqdm.notebook import tqdm

import torchvision.transforms.functional as TF
from torchvision import transforms
PIL = transforms.ToPILImage()
TTen = transforms.ToTensor()
from PIL import Image


# rotate ============================================================================
def normalize(tensor):
    return (tensor - torch.min(tensor))/(torch.max(tensor) - torch.min(tensor))

def rotate(img, degree):
    #img shape 2*128*128
    #2*2 2*1*128*128 -> 2*1*128*128
    theta = torch.tensor(degree/180*np.pi)
    rot_m = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]])
    img = torch.einsum("ab, bcde -> acde",(rot_m, img.unsqueeze(1))).squeeze(1)
    mmin = torch.min(img)
    mmax = torch.max(img)
    img = normalize(img).data.numpy()
    x = TTen(TF.rotate(Image.fromarray(np.uint8(img[0]*255)), degree, expand=True, fill=(0,)))
    y = TTen(TF.rotate(Image.fromarray(np.uint8(img[1]*255)), degree, expand=True, fill=(0,)))
    rot_img = torch.cat([x, y], dim = 0)
    rot_img[rot_img!=0] = normalize(rot_img[rot_img!=0])
    rot_img[rot_img!=0] = rot_img[rot_img!=0]*(mmax - mmin) + mmin
    return rot_img

def pad_after_rot(rot_um_img, target_dim=92):
    dim_diff = (target_dim - rot_um_img.shape[-1]) // 2
    pad_to_dim = (dim_diff, dim_diff, dim_diff, dim_diff)
    return F.pad(rot_um_img, pad_to_dim, 'constant', 0)

direc = "/global/cscratch1/sd/roseyu/Eliza/TF-net/Data/data_64/sample_"
for i in range(9870):
    degree = (15*(i-7000))%360
    img = torch.load(direc + str(i) + ".pt")#+(torch.rand(1, 2, 1, 1)*4-2
    rot_img = torch.cat([rotate(img[j], degree).unsqueeze(0) for j in range(img.shape[0])], dim = 0)
    rot_img_padded = pad_after_rot(rot_img)
    #break
    torch.save(rot_img_padded, "/global/cscratch1/sd/roseyu/Eliza/TF-net/Data/rot_64/sample_" + str(i) + ".pt")

# UM ===============================================================================
def sample_within_spherical():
    t = 2 * np.pi * np.random.uniform()
    u = np.random.uniform() + np.random.uniform()
    r = 2 - u if u > 1 else u
    return [r * np.cos(t), r * np.sin(t)]

def sample_n_within_spherical(n=1):
    return np.array([sample_within_spherical() for i in range(n)]).transpose() 

def uniform_motion(img, unit_vector):
    return img + torch.FloatTensor(unit_vector).repeat(img.shape[0], img.shape[-1]**2).view(img.shape)
  
direc = "/global/cscratch1/sd/roseyu/Eliza/TF-net/Data/rot_64/sample_"
for i in range(9870):
    um_vector = sample_n_within_spherical()
    img = torch.load(direc + str(i) + ".pt")#+(torch.rand(1, 2, 1, 1)*4-2
    rot_um_img = torch.cat([uniform_motion(img, um_vector) for j in range(img.shape[0])], dim = 0)
    #break
    torch.save(rot_um_img, "/global/cscratch1/sd/roseyu/Eliza/TF-net/Data/rot_um_64/sample_" + str(i) + ".pt")

# magnitude ============================================================================
def magnitude(img, scalar):
    return img * scalar

direc = "/global/cscratch1/sd/roseyu/Eliza/TF-net/Data/rot_um_64/sample_"
for i in range(9870):
    mag_scalar = 2 - np.random.uniform(0, 2) # ensure 0 is not choosen
    img = torch.load(direc + str(i) + ".pt")#+(torch.rand(1, 2, 1, 1)*4-2
    rot_um_mag_img = magnitude(img, mag_scalar)
    #break
    torch.save(rot_um_mag_img, "/global/cscratch1/sd/roseyu/Eliza/TF-net/Data/rot_um_mag_64/sample_" + str(i) + ".pt")