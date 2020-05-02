import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.utils import data
import warnings
from .train import train_epoch, eval_epoch, test_epoch, Dataset

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.chdir("/global/cscratch1/sd/rwang2/Equivariance/Anisotropic")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
basis = torch.load("kernel_basis.pt")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Ani_layer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, activation = False):
        super(Ani_layer, self).__init__()
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.activation = activation
        self.kernel_size = kernel_size
        self.radius = (kernel_size - 1)//2
        self.pad_size = (kernel_size - 1)//2
        self.num_weights = self.radius*4
        self.basis = basis[:self.num_weights, :, 15-self.radius:self.radius-15, 15-self.radius:self.radius-15]
        self.params = self.init_params()
    
        self.bias_term = nn.Parameter(torch.ones(1, self.output_channels, 2, 1, 1)/100)
        
        initial_kernel = torch.einsum("abcd, cdefgh -> abcdefgh",  (self.params, self.basis))
        stds = torch.std(initial_kernel[initial_kernel != 0.0])
        
        self.scaler = np.sqrt(0.6/(np.sqrt(input_channels) * self.kernel_size**2 * stds))
        
        self.params = nn.Parameter(self.params * self.scaler.unsqueeze(0).unsqueeze(0).unsqueeze(-1))
        self.basis = self.basis * self.scaler.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.b = nn.Parameter(torch.tensor(0.01))
        
        self.initial_params = self.params.clone()
        self.initial_kernel = torch.einsum("abcd, cdefgh -> abefgh",  (self.params, self.basis)).clone()
        
    def init_params(self):
        return torch.randn(self.output_channels, self.input_channels, self.num_weights, 4)
    
    def get_kernel(self, params, basis):
        # Compute Kernel: Kernel shape (output_channels, input_channels, kernel_size, kernel_size, 2, 2)
        kernel = torch.einsum("abcd, cdefgh -> abefgh",  (self.params, self.basis.to(device)))

        # Reshape
        kernel = kernel.transpose(-2, -3).transpose(-3, -4).transpose(-4, -5)
        kernel = kernel.reshape(kernel.shape[0]*2,  kernel.shape[2], self.kernel_size, self.kernel_size, 2)
        kernel = kernel.transpose(-1, -2).transpose(-2, -3)
        kernel = kernel.reshape(kernel.shape[0], kernel.shape[1]*2, self.kernel_size, self.kernel_size)

        return kernel
    
    def forward(self, xx):

        kernel = self.get_kernel(self.params, self.basis)

        # Conv2d
        out = F.conv2d(xx, kernel, padding = self.pad_size) #, bias = self.bias_term
        out = out.reshape(out.shape[0], out.shape[1]//2, 2, xx.shape[-2], xx.shape[-1])
        #out += self.bias_term
        
        # Activation Function
        norm = torch.sqrt(out[:,:,0,:,:]**2 + out[:,:,1,:,:]**2).unsqueeze(2)
        out = out*torch.sin(norm)/norm    
        
        out = out.reshape(out.shape[0], -1, out.shape[-2], out.shape[-1])
        return out
    
    
class Resblock(nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size, activation, skip = True):
        super(Resblock, self).__init__()
        self.layer1 = Ani_layer(input_channels, hidden_dim, kernel_size, activation = activation)
        self.layer2 = Ani_layer(hidden_dim, hidden_dim, kernel_size, activation = activation)
        self.skip = skip
        
    def forward(self, x):
        out = self.layer1(x)
        if self.skip:
            out = self.layer2(out) + x
        else:
            out = self.layer2(out)
        return out


class rot_cnn(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, activation):
        super(rot_cnn, self).__init__()
        layers = [Ani_layer(input_channels, 64, kernel_size, activation=activation)]
        layers += [Resblock(64, 64, kernel_size, activation=activation, skip=True) for i in range(3)]
        layers += [Resblock(64, 128, kernel_size, activation=activation, skip=False)] + [Resblock(128, 128, kernel_size, activation=activation, skip=True) for i in range(3)]
        layers += [Resblock(128, 256, kernel_size, activation=activation, skip=False)] + [Resblock(256, 256, kernel_size, activation=activation, skip=True) for i in range(5)]
        layers += [Resblock(256, 512, kernel_size, activation=activation, skip=False)] + [Resblock(512, 512, kernel_size, activation=activation, skip=True) for i in range(2)]
        layers += [Ani_layer(512, 1, kernel_size, False)]
        self.layers = nn.Sequential(*layers)

    def forward(self, xx):
        return self.layers(xx)     


train_direc = "/global/cscratch1/sd/rwang2/TF-net/Data/data_64/sample_"
test_direc = "/global/cscratch1/sd/rwang2/TF-net/Data/data_64/sample_"
kernel_size = 3
learning_rate = 1e-05
output_length = 3
batch_size = 8
input_length = 20
train_indices = list(range(0, 6000))
valid_indices = list(range(6000, 7700))
test_indices = list(range(7700, 9470))

train_set = Dataset(train_indices, input_length, 40, output_length, train_direc, True)
valid_set = Dataset(valid_indices, input_length, 40, 6, train_direc, True)
train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
valid_loader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8)

print("Initializing...")
model = nn.DataParallel(rot_cnn(input_channels=input_length, output_channels=1, kernel_size=kernel_size, activation="sin").to(device))
print("Done")

optimizer = torch.optim.Adam(model.parameters(), learning_rate,betas=(0.9, 0.999), weight_decay=4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
loss_fun = torch.nn.MSELoss()

train_mse = []
valid_mse = []
test_mse = []

min_mse = 10
for i in range(30):
    start = time.time()
    scheduler.step()

    model.train()
    train_mse.append(train_epoch(train_loader, model, optimizer, loss_fun))
    model.eval()
    mse, _, _ = eval_epoch(valid_loader, model, loss_fun)
    valid_mse.append(mse)

    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1] 
        best_model = model
        torch.save(model, "model.pth")
    end = time.time()
    if len(train_mse) > 30 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5]):
        break
    print(train_mse[-1], valid_mse[-1], round((end-start)/60,5), format(get_lr(optimizer), "5.2e"))

print("*******", input_length, min_mse, "*******")


best_model = torch.load("model.pth")
loss_fun = torch.nn.MSELoss()
test_set = Dataset(test_indices, input_length, 40, 60, test_direc, True)
test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 16)
valid_mse, preds, trues, loss_curve = test_epoch(test_loader, best_model, loss_fun)

torch.save({"preds": preds[:7],
            "trues": trues[:7],
            "loss_curve": loss_curve,
            "train_loss": train_mse,
            "valid_loss": valid_mse},
           "results.pt")

"""
class Ani_layer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, activation = False):
        super(Ani_layer, self).__init__()
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.activation = activation
        self.kernel_size = kernel_size
        self.radius = (kernel_size - 1)//2
        self.pad_size = (kernel_size - 1)//2
        self.basis = basis[:self.radius, :, 15-self.radius:self.radius-15, 15-self.radius:self.radius-15]
        
        self.params = self.init_params()
        scaler = np.sqrt(0.6/(np.sqrt(input_channels)*self.kernel_size**2*torch.std(torch.einsum("abcd, cdefgh -> abefgh",  (self.params, self.basis)))))
        self.params = nn.Parameter(self.params*scaler)
        self.basis = self.basis*scaler
        
    def init_params(self):
        return torch.randn((self.output_channels, self.input_channels, self.radius, 4))
    
    def input_reshape(self, xx):
        out = F.unfold(xx, kernel_size = self.kernel_size).transpose(1,2)
        out = out.reshape(xx.shape[0], -1, self.input_channels*2, self.kernel_size, self.kernel_size)
        out = out.reshape(xx.shape[0], xx.shape[-2]-2*self.pad_size, xx.shape[-1]-2*self.pad_size, 
                      self.input_channels*2, self.kernel_size, self.kernel_size)
        out = out.reshape(xx.shape[0], xx.shape[-2]-2*self.pad_size, xx.shape[-1]-2*self.pad_size, 
                      self.input_channels, 2, self.kernel_size, self.kernel_size).transpose(-2,-3).transpose(-1,-2).unsqueeze(-2)
        return out
    
    def forward(self, xx):
        #kernel shape: (output_channels, input_channels, kernel_size, kernel_size, 2, 2)
        
        kernel = torch.einsum("abcd, cdefgh -> abefgh",  (self.params, self.basis.to(device)))
        
        # Padding
        inp_xx = F.pad(xx, (self.pad_size, self.pad_size, self.pad_size, self.pad_size))
        
        # Unfold & Reshape
        inp_xx = self.input_reshape(inp_xx)
        
        # Output Tensor
        #output = torch.cat([torch.einsum("acdefgh, xdefhy -> axcgy", (inp_xx[:,i], kernel)).squeeze(-2).unsqueeze(1) for i in range(inp_xx.shape[1])], dim =1)
        output = torch.einsum("abcdefgh, xdefhy -> axbcgy", (inp_xx, kernel)).squeeze(-2)
        
        if self.activation:
            norm = torch.sqrt(output[:,:,:,:,0]**2 + output[:,:,:,:,1]**2).unsqueeze(-1)
            output = output*torch.tanh(norm)/norm
            output = output.reshape(xx.shape[0], self.output_channels*2, xx.shape[-2], xx.shape[-1])
        else:
            output = output.reshape(xx.shape[0], self.output_channels*2, xx.shape[-2], xx.shape[-1])
        return output

"""