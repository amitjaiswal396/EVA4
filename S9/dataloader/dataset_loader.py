import torch
from torchvision import datasets, transforms
import albumentations
import albumentations.augmentations.transforms as aug_transform
import albumentations.pytorch.transforms as torch_transform
import dataloader.data_transform as data_transform

# function to load train and test data
def get_dataloader(is_train, cuda):
    if is_train:
        transform = data_transform.albumentations_transforms(p=1.0, is_train=True)
    else:
        transform = data_transform.albumentations_transforms(p=1.0, is_train=False)

    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=16)

    dataset = datasets.CIFAR10(root='./data', train=is_train, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_args)
    return dataloader
