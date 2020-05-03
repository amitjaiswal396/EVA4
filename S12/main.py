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
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from dataloader import dataset_loader
from evaluation import test
from training import train
from learning_rate_finder import lr_rangefinder
from models import resnet

from cam.gradcam import GradCAM
from cam.utils import visualize_cam, Normalize



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
trainloader = dataset_loader.get_dataloader(True, cuda)
## getting test data loader
testloader = dataset_loader.get_dataloader(False, cuda)
