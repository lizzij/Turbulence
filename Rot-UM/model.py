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


train_direc = "/global/cscratch1/sd/roseyu/Eliza/TF-net/Data/data_64/sample_"
test_direc = "/global/cscratch1/sd/roseyu/Eliza/TF-net/Data/data_64/sample_"
kernel_size = 3
num_layers = 2
hidden_dim = 128
learning_rate = 1e-05
output_length = 3
batch_size = 1
input_length = 20
# train_indices = list(range(0, 6000))
# valid_indices = list(range(6000, 8000))
# test_indices = list(range(8000, 9870))

# train_indices = list(range(0, 600))
# valid_indices = list(range(600, 800))
# test_indices = list(range(8000, 8200))

# train_indices = list(range(0, 60))
# valid_indices = list(range(60, 80))
# test_indices = list(range(8000, 8020))

train_indices = list(range(0, 6))
valid_indices = list(range(6, 8))
test_indices = list(range(8000, 8002))

# train_indices = list(range(8000, 8006))
# valid_indices = list(range(8006, 8008))
# test_indices = list(range(8008, 8010))


train_set = Dataset(train_indices, input_length, 40, output_length, train_direc, True)
valid_set = Dataset(valid_indices, input_length, 40, 6, train_direc, True)
train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
valid_loader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8)

print("Initializing...")
model = rot_um_cnn(activation = "relu", input_channels = input_length, hidden_dim = hidden_dim, num_layers = num_layers, output_channels = 1, kernel_size = kernel_size).to(device)
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