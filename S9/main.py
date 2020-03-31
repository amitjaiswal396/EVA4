# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from tqdm import tqdm

import train_test as tt
from Models import resnet

from dataloader.dataset_loader import get_dataloader
from dataloader import data_transform as data_transform
from gradcam import GradCAM, visualize_cam

from torchvision.utils import make_grid, save_image



##############################################################################
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# setting up random seed and processing device
SEED = 1
cuda = torch.cuda.is_available()
if cuda:
    device = "cuda"
    torch.cuda.manual_seed(SEED)
else:
    device = "cpu"
    torch.manual_seed(SEED)

## getting training data loader
#trainloader = Datasetloader(True, cuda).trainloader()

## getting test data loader
#testloader = Datasetloader(False, cuda).testloader()

## getting training data loader
trainloader = get_dataloader(True, cuda)

## getting test data loader
testloader = get_dataloader(False, cuda)


##############################################################################
#get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
images.size()
#plt.imshow(images[0].permute(1, 2, 0))

#show images
plt.imshow(images[4].permute(1,2,0))

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    
# show images
show(torchvision.utils.make_grid(images))


##############################################################################
mod = resnet.ResNet18().to(device)
summary(mod, input_size=(3, 32, 32))


##############################################################################
import torch
from tqdm import tqdm

# model training
def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    train_acc = []

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        max_prob = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += max_prob.eq(target.view_as(max_prob)).sum().item()
        processed += len(data)
        pbar.set_description(desc= f'epoch={epoch} Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)



##############################################################################
# loss function
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(mod.parameters(), lr=0.01, momentum=0.9)

# Model Training and Evaluation
#for epoch in range(0, 1):
#    tt.train(mod, device, trainloader, optimizer, epoch, criterion)
#    tt.test(mod, device, testloader, criterion)
    
    
##############################################################################
dataiter_test = iter(testloader)
#print(dataiter)
test_images, test_labels = dataiter_test.next()
test_images.size()
test_images

images = []
target_layers = ['layer4']

for layers, module in mod.to(device).eval().named_modules():
    if layers in target_layers:
        gcam = GradCAM(mod.to(device).eval(), module)
        mask, _ = gcam(test_images[0:1, :, :, :])
        print(test_images.size())
        heatmap, result = visualize_cam(mask, test_images[0])        
        images.extend([test_images[0].cpu(), heatmap, result])

grid_image = make_grid(images, nrow=5) 
transforms.ToPILImage()(grid_image)
print("dd")
    
    








