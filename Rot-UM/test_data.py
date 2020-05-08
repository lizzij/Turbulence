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
from tqdm.notebook import tqdm
#import shutil
#shutil.rmtree('/global/cscratch1/sd/rwang2/Data/Subsample_128')

import torchvision.transforms.functional as TF
from torchvision import transforms
PIL = transforms.ToPILImage()
TTen = transforms.ToTensor()
from PIL import Image

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

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def uniform_motion(img, unit_vector):
    return img + torch.FloatTensor(unit_vector).repeat(img.shape[0], img.shape[-1]**2).view(img.shape)

print('starting')

direc = "/global/cscratch1/sd/roseyu/Eliza/TF-net/Data/data_64/sample_"
for i in tqdm(range(8421, 9870)):
    um_vector = sample_spherical(1, 2)
    img = torch.load(direc + str(i) + ".pt")#+(torch.rand(1, 2, 1, 1)*4-2
    um_img = torch.cat([uniform_motion(img, um_vector) for j in range(img.shape[0])], dim = 0)
    #break
    torch.save(um_img, "/global/cscratch1/sd/roseyu/Eliza/TF-net/Data/um_64/sample_" + str(i) + ".pt")

print('um done!')

direc = "/global/cscratch1/sd/roseyu/Eliza/TF-net/Data/rot_64/sample_"
for i in tqdm(range(8600, 9870)):
    um_vector = sample_spherical(1, 2)
    img = torch.load(direc + str(i) + ".pt")#+(torch.rand(1, 2, 1, 1)*4-2
    rot_um_img = torch.cat([uniform_motion(img, um_vector) for j in range(img.shape[0])], dim = 0)
    #break
    torch.save(rot_um_img, "/global/cscratch1/sd/roseyu/Eliza/TF-net/Data/rot_um_64/sample_" + str(i) + ".pt")
    
print('rot um done!')