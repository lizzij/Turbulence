import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
from model import rot_um_cnn
from train import train_epoch, eval_epoch, test_epoch, Dataset, get_lr
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_direc = "/global/cscratch1/sd/roseyu/Eliza/TF-net/Data/rot_um_64/sample_"
idx = str(1)
num_layers = 2
kernel_size = 3
hidden_dim = 128
learning_rate = 0.001
min_mse = 1
output_length = 4
batch_size = 1
input_length = 25
# train_indices = list(range(0, 6000))
# valid_indices = list(range(6000, 8000))
# test_indices = list(range(0, 2000))

# train_indices = list(range(0, 600))
# valid_indices = list(range(600, 800))
# test_indices = list(range(8000, 8200))

# train_indices = list(range(0, 60))
# valid_indices = list(range(60, 80))
# test_indices = list(range(80, 100))

train_indices = list(range(0, 6))
valid_indices = list(range(6, 8))
test_indices = list(range(8, 10))

# train_indices = list(range(8000, 8006))
# valid_indices = list(range(8006, 8008))
# test_indices = list(range(8008, 8010))

train_set = Dataset(train_indices, input_length, 40, output_length, train_direc, True)
valid_set = Dataset(valid_indices, input_length, 40, 6, train_direc, True)
train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 8)

print("Initializing...")
model = rot_um_cnn(activation = "relu", input_channels = input_length, hidden_dim = hidden_dim, num_layers = num_layers, output_channels = 1, kernel_size = kernel_size).to(device)
print("Done")

optimizer = torch.optim.Adam(model.parameters(), learning_rate,betas=(0.9, 0.999), weight_decay=4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=0.9)
loss_fun = torch.nn.MSELoss()


train_mse = []
valid_mse = []
test_mse = []

# train model
for i in range(100):
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
        torch.save(model, "Rot-UM-CNN"+idx+".pth")
    end = time.time()
    if (len(train_mse) > 50 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
            break
    print(i+1,train_mse[-1], valid_mse[-1], round((end-start)/60,5), format(get_lr(optimizer), "5.2e"), idx)


# # load from saved
# best_model = torch.load("Rot-UM-CNN1.pth").to(device)

print('\n-------------test_mse-------------')

# test_direc = "/global/cscratch1/sd/roseyu/Eliza/TF-net/Data/data_64/sample_"
# loss_fun = torch.nn.MSELoss()
# test_set = Dataset(test_indices, input_length, 40, 10, test_direc, True)
# test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
# test_mse, preds, trues, loss_curve = test_epoch(test_loader, best_model, loss_fun)

# print('Rot-UM-CNN:',test_mse)

# torch.save({"preds": preds,
#             "trues": trues,
#             "test_mse":test_mse,
#             "loss_curve": loss_curve}, 
#             "Rot-UM-CNN"+idx+".pt")

# test_direc = "/global/cscratch1/sd/roseyu/Eliza/TF-net/Data/um_64/sample_"
# test_set = Dataset(test_indices, input_length, 40, 10, test_direc, True)
# test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
# test_mse, preds, trues, loss_curve = test_epoch(test_loader, best_model, loss_fun)

# torch.save({"preds": preds,
#             "trues": trues,
#             "test_mse":test_mse,
#             "loss_curve": loss_curve}, 
#             "Rot-UM-CNN-UM"+idx+".pt")

# print('Rot-UM-CNN-UM:',test_mse)

# test_direc = "/global/cscratch1/sd/roseyu/Eliza/TF-net/Data/rot_64/sample_"
# test_set = Dataset(test_indices, input_length, 40, 10, test_direc, True)
# test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
# test_mse, preds, trues, loss_curve = test_epoch(test_loader, best_model, loss_fun)

# torch.save({"preds": preds,
#             "trues": trues,
#             "test_mse":test_mse,
#             "loss_curve": loss_curve}, 
#             "Rot-UM-CNN-Rot"+idx+".pt")

# print('Rot-UM-CNN-Rot:',test_mse)

# test_direc = "/global/cscratch1/sd/roseyu/Eliza/TF-net/Data/rot_um_64/sample_"
# test_set = Dataset(test_indices, input_length, 40, 10, test_direc, True)
# test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
# test_mse, preds, trues, loss_curve = test_epoch(test_loader, best_model, loss_fun)

# torch.save({"preds": preds,
#             "trues": trues,
#             "test_mse":test_mse,
#             "loss_curve": loss_curve}, 
#             "Rot-UM-CNN-Org-Rot-UM"+idx+".pt")

# print('Rot-UM-CNN-Org-Rot-UM:',test_mse)

test_direc = "/global/cscratch1/sd/roseyu/Eliza/TF-net/Data/rot_um_64/sample_"
test_set = Dataset(test_indices, input_length, 40, 10, test_direc, True)
test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
test_mse, preds, trues, loss_curve = test_epoch(test_loader, best_model, loss_fun)

torch.save({"preds": preds,
            "trues": trues,
            "test_mse":test_mse,
            "loss_curve": loss_curve}, 
            "Rot-UM-CNN-Rot-UM-Rot-UM"+idx+".pt")

print('Rot-UM-CNN-Rot-UM-Rot-UM:',test_mse)