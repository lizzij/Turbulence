import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.utils import data
import warnings
from train import train_epoch, eval_epoch, test_epoch, Dataset

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
basis = torch.load("kernel_basis.pt")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

    

class Ani_layer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, um_dim = 2, activation = False):
        super(Ani_layer, self).__init__()
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.activation = activation
        self.kernel_size = kernel_size
        self.um_dim = um_dim
        self.radius = (kernel_size - 1)//2
        self.pad_size = (kernel_size - 1)//2
        self.num_weights = self.radius*4
        self.basis = basis[:self.num_weights, :, 15-self.radius:self.radius-15, 15-self.radius:self.radius-15]
        self.params = self.init_params()  
        
        self.bias_term = nn.Parameter(torch.ones(1, self.output_channels, 2, 1, 1)/100)
        
        initial_kernel = torch.einsum("abcd, cdefgh -> abcdefgh",  (self.params, self.basis))
        stds = torch.std(initial_kernel[initial_kernel != 0.0])
        
        #print(torch.std(initial_kernel, dim = (0,1,4,5,6,7)))
        
        self.scaler = np.sqrt(0.6/(np.sqrt(input_channels) * self.kernel_size**2 * stds))
        
        self.params = nn.Parameter(self.params * self.scaler)#.unsqueeze(0).unsqueeze(0).unsqueeze(-1))
        self.basis = self.basis * self.scaler#.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        self.b = nn.Parameter(torch.tensor(0.01))
        
        self.initial_params = self.params.clone()
        self.initial_kernel = torch.einsum("abcd, cdefgh -> abefgh",  (self.params, self.basis)).clone()
        
    def init_params(self):
        return torch.randn((self.output_channels, self.input_channels, self.num_weights, 4))
    
    
    def get_kernel(self, params, basis):
        # Compute Kernel: Kernel shape (output_channels, input_channels, kernel_size, kernel_size, 2, 2) 
        kernel = torch.einsum("abcd, cdefgh -> abefgh",  (self.params, self.basis.to(device)))
        
        # Reshape
        kernel = kernel.transpose(-2, -3).transpose(-3, -4).transpose(-4, -5)       
        kernel = kernel.reshape(kernel.shape[0]*2,  kernel.shape[2], self.kernel_size, self.kernel_size, 2)
        kernel = kernel.transpose(-1, -2).transpose(-2, -3)
        kernel = kernel.reshape(kernel.shape[0], kernel.shape[1]*2, self.kernel_size, self.kernel_size)

        return kernel
    
    def unfold(self, xx):
        out = F.pad(xx, ((self.pad_size, self.pad_size)*2), mode='replicate')
        out = F.unfold(out, kernel_size = self.kernel_size)
        out = out.reshape(out.shape[0], self.input_channels * self.um_dim, self.kernel_size, self.kernel_size, out.shape[-1])
        out = out.reshape(out.shape[0], self.input_channels * self.um_dim, self.kernel_size, self.kernel_size, xx.shape[-2], xx.shape[-1])
        return out
    
    def subtract_mean(self, xx):
        out = xx.reshape(xx.shape[0], self.input_channels, self.um_dim, self.kernel_size, self.kernel_size, xx.shape[-2], xx.shape[-1])
        avgs = out.mean((1,3,4), keepdim=True)
        out -= avgs
        out = out.reshape(out.shape[0], self.input_channels * self.um_dim, self.kernel_size, self.kernel_size, xx.shape[-2], xx.shape[-1]).transpose(2,4).transpose(-1,-2)
        out = out.reshape(out.shape[0], self.input_channels * self.um_dim, xx.shape[-2]*self.kernel_size, xx.shape[-1], self.kernel_size)
        out = out.reshape(out.shape[0], self.input_channels * self.um_dim, xx.shape[-2]*self.kernel_size, xx.shape[-1]*self.kernel_size)
        return out, avgs.squeeze(3).squeeze(3)
    
    
    def add_mean(self, out, avgs):
        out += avgs
        out = out.reshape(out.shape[0], -1, out.shape[-2], out.shape[-1])
        return out
    
    def forward(self, xx, add_mean = True):       
        kernel = self.get_kernel(self.params, self.basis)
        
        xx = self.unfold(xx)
        xx, avgs = self.subtract_mean(xx)

        # Conv2d
        out = F.conv2d(xx, kernel, stride = self.kernel_size) #, bias = self.bias_term
        out = out.reshape(out.shape[0], out.shape[1]//2, 2, out.shape[-2], out.shape[-1])
        out += self.bias_term
        
        # Activation Function
        if self.activation:
            print("***")
            if self.activation == "sin":
                norm = torch.sqrt(out[:,:,0,:,:]**2 + out[:,:,1,:,:]**2).unsqueeze(2)
                out = out*torch.sin(norm)**2/norm

            elif self.activation == "relu":
                norm = torch.sqrt(out[:,:,0,:,:]**2 + out[:,:,1,:,:]**2).unsqueeze(2).repeat(1,1,2,1,1) 
                out = out/norm
                norm2 = norm - self.b
                out[norm2 <= 0.] = 0.    

            elif self.activation == "leakyrelu":
                norm = torch.sqrt(out[:,:,0,:,:]**2 + out[:,:,1,:,:]**2).unsqueeze(2).repeat(1,1,2,1,1) 
                out = out/norm
                norm2 = norm - self.b
                out[norm2 <= 0.] = out[norm2 <= 0.]*0.1

            elif self.activation == "squash":
                norm = torch.sqrt(out[:,:,0,:,:]**2 + out[:,:,1,:,:]**2).unsqueeze(2)
                out = out/norm*(norm**2/(norm**2+1))
        if add_mean:
            out = self.add_mean(out, avgs)
        return out
    
class rot_um_cnn(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_dim, num_layers, kernel_size, activation = False):
        super(rot_um_cnn, self).__init__()
        layers = [Ani_layer(input_channels, hidden_dim, kernel_size, activation=activation)] + \
                 [Ani_layer(hidden_dim, hidden_dim, kernel_size, activation=activation) for i in range(num_layers - 2)] + \
                 [Ani_layer(hidden_dim, output_channels, kernel_size)]
        self.layers = nn.Sequential(*layers)

    def forward(self, xx):
        return self.layers(xx)