import torch
from torchvision import datasets
import dataloader.data_transform as data_transform
from config import config
import os

# function to load train and test data
def get_dataloader(is_train, cuda):
    if is_train:
        transform = data_transform.albumentations_transforms(p=1.0, is_train=True)
    else:
        transform = data_transform.albumentations_transforms(p=1.0, is_train=False)

    dataloader_args = dict(shuffle=True, batch_size=512, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=16)

    dataset = datasets.CIFAR10(root='./data', train=is_train, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_args)
    return dataloader



def imagenet_dataloader(data_split, cuda):
    data_dir = config.DATA_DIR+'/tiny-imagenet-200/'
    num_workers = {'train':100, 'val':0, 'test':0}
    dataloader_args = dict(shuffle=True, batch_size=512, num_workers=num_workers[data_split], pin_memory=True) if cuda else dict(shuffle=True, batch_size=16)
    
    transform = data_transform.pytorch_transform(data_split)
        
    dataset = datasets.ImageFolder(os.path.join(data_dir, data_split), transform[data_split])
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_args)                             
    
    return dataloader
