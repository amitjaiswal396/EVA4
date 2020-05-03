from albumentations import (Compose, HorizontalFlip, Normalize, Rotate, HueSaturationValue, Cutout)
from albumentations.pytorch import ToTensor
import numpy as np	
import torch
from torchvision import datasets, transforms


def albumentations_transforms(p=1.0, is_train=False):
	mean = np.array([0.4914, 0.4822, 0.4465])
	std = np.array([0.2023, 0.1994, 0.2010])
	transforms_list = []
	if is_train:
		transforms_list.extend([HueSaturationValue(p=0.25),
			                    HorizontalFlip(p=0.5),
			                    Rotate(limit=15),
			                    Cutout(),])
                                
	transforms_list.extend([Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
		                    ToTensor(),])
	transforms = Compose(transforms_list, p=p)
	return lambda img:transforms(image=np.array(img))["image"]


def pytorch_transform(is_train=False):
    mean = np.array([0.4802, 0.4481, 0.3975])
    std = np.array([0.2302, 0.2265, 0.2262])
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    }
    return data_transforms