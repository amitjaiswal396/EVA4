import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from dataloader import dataset_loader
from evaluation import test
# from google.colab import drive
from model import model
from training import train

# Mounting Google Drive
# drive.mount("/content/gdrive/", force_remount=True)

# setting project workspace
# sys.path.append('/content/gdrive/My Drive/EVA4_Session_8')

# !ls /content/gdrive/My\ Drive/EVA4_Session_8

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# setting up random seed and processing\ device
SEED = 1
cuda = torch.cuda.is_available()
if cuda:
    device = "cuda"
    torch.cuda.manual_seed(SEED)
else:
    device = "cpu"
    torch.manual_seed(SEED)

# getting training data loader
trainloader = dataset_loader.get_dataloader(True, cuda)

# getting test data loader
testloader = dataset_loader.get_dataloader(False, cuda)

# !pip install torchsummary
resnet18 = model.ResNet18().to(device)
print(summary(resnet18, input_size=(3, 32, 32)))

# loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

# Model Training and Evaluation
for epoch in range(0, 25):
    train.train(resnet18, device, trainloader, optimizer, epoch, criterion)
    test.test(resnet18, device, testloader, criterion)
