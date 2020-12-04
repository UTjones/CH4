import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np

import matplotlib.pyplot as plt

import torch.optim.lr_scheduler #import OneCycleLR

import json
import math


import torch.nn as nn
import torch.nn.functional as F

import torch.optim 



with open('11_grid_bins.json', 'r') as fp:
    data = json.load(fp)



res = list(data.keys())[0] 
num_grids = len(data)
train_num = math.floor(0.7*num_grids)
test_num = num_grids - train_num



grid_list = []
for key in data.keys():
    grid_list.append([data[key][0], data[key][1]])

np.random.shuffle(grid_list)

height, width = np.shape(grid_list[0][0])[0], np.shape(grid_list[0][0])[1]

train_grids = torch.zeros(train_num, height, width)
train_labels = torch.zeros(train_num)

test_grids = torch.zeros(test_num, height, width)
test_labels = torch.zeros(test_num)

for i in range(num_grids):
    if i < train_num:
        train_grids[i] = torch.FloatTensor(grid_list[i][0])
        train_labels[i] = float(grid_list[i][1])
    else:
        new_i = i - train_num
        test_grids[new_i] = torch.FloatTensor(grid_list[i][0])
        test_labels[new_i] = float(grid_list[i][1])
        




class cgNet(nn.Module):
   
    def __init__(self):
        super(cgNet, self).__init__()
        
        #first block
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels= 32, 
                               kernel_size = 3,padding = 1 ,
                               padding_mode = 'replicate')
        self._bn_1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels= 32, 
                               kernel_size = 3,padding = 1 ,
                               padding_mode = 'replicate')
        self._bn_2 = nn.BatchNorm2d(32)
        self.max_pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(p = 0.3, inplace=False)
        
        
        #second block
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels= 64, 
                               kernel_size = 3,padding = 1 ,
                               padding_mode = 'replicate')
        self._bn_3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels= 64, 
                               kernel_size = 3,padding = 1 ,
                               padding_mode = 'replicate')
        self._bn_4 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(p = 0.3, inplace=False)
        
        #third block
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels= 128, 
                               kernel_size = 3,padding = 1 ,
                               padding_mode = 'replicate')
        self._bn_5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(in_channels = 128, out_channels= 128, 
                               kernel_size = 3,padding = 1 ,
                               padding_mode = 'replicate')
        self._bn_6 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(p = 0.3, inplace=False)
        
        #fourth block
        self.fc1 = nn.Linear(128,512)
        
        self.fc2 = nn.Linear(512, 1)
        


        
    def forward(self, x):
        #First block
        x = F.relu(self._bn_1(self.conv1(x)))
        x = F.relu(self._bn_2(self.conv2(x)))
        x = self.max_pool(x)
        # Uncomment these dropout layers if you want to train with dropout
	#x = self.dropout1(x)
        
        #second block
        x = F.relu(self._bn_3(self.conv3(x)))
        x = F.relu(self._bn_4(self.conv4(x)))
        x = self.max_pool(x)
        #x = self.dropout2(x)
        
        #third block
        x = F.relu(self._bn_5(self.conv5(x)))
        x = F.relu(self._bn_6(self.conv6(x)))
        x = self.max_pool(x)
        #x = self.dropout3(x)
        
        #fourth block
        #x = F.adaptive_avg_pool2d(x, (1, 1))
        
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        #x = self.dropout3(x)
        x = self.fc2(x)
        
        return x

network = cgNet()

loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(network.parameters(), lr=0.0001) #, momentum=0.5)
one_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                steps_per_epoch=len(train_grids), epochs=10, max_lr=0.1)
cycle_on = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device {device}")

network = network.to(device)

batch_size = 32
dataset = torch.utils.data.TensorDataset(train_grids, train_labels)
loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
datatest = torch.utils.data.TensorDataset(test_grids, test_labels)
testset = torch.utils.data.DataLoader(datatest, batch_size = batch_size, shuffle = False)


def test_acc(testset):
    test_dict = {'pred': [], 'label': []}
    total = 0
    with torch.no_grad():
        for images, labels in testset:

            images = torch.reshape(images, (images.size(0), 1, 11, 11))
            

            outputs = network(images)
            outputs = outputs.flatten()
            total += torch.sum(torch.sub(outputs, labels))
            test_dict['pred'].extend(outputs.tolist())
            test_dict['label'].extend(labels.tolist())
            
        acc = (total/test_num).item()
        

        return(round(acc, 4), test_dict)



EPOCHS = 20
val_dict = {'acc' : [], 'loss': [], 'epoch': []}
for epoch in range(EPOCHS):
    total_loss = 0
    count = 0
    for images, labels in loader:
        images = torch.reshape(images, (images.size(0), 1, 11, 11))
        
        network.zero_grad()

        outputs = network(images)
        outputs = outputs.flatten()

        loss = loss_function(outputs, labels)

        total_loss += loss.cpu().detach().item()
        count += 1

        loss.backward()
        optimizer.step()
        one_scheduler.step()


    total_loss = float(total_loss)/float(count)
    

    acc, test_dict = test_acc(testset)
    val_dict['acc'].append(acc)
    val_dict['loss'].append(total_loss)
    val_dict['epoch'].append(epoch)

print(val_dict)

plt.plot(range(1, len(test_dict['pred']) + 1), test_dict['pred'], '.-', label='pred')
plt.plot(range(1, len(test_dict['label']) + 1), test_dict['label'], '.-', label='act')
plt.legend()
plt.show()
