from albumentations import (Compose, HorizontalFlip, Normalize, Rotate, HueSaturationValue, Cutout)
from albumentations.pytorch import ToTensor
import numpy as np	

def albumentations_transforms(p=1.0, is_train=False):
	mean = np.array([0.4914, 0.4822, 0.4465])
	std = np.array([0.2023, 0.1994, 0.2010])
	transforms_list = []
	if is_train:
		transforms_list.extend([HueSaturationValue(p=0.25),
			                    HorizontalFlip(p=0.5),
			                    Rotate(limit=15),
			                    Cutout()])
                                
	transforms_list.extend([Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
		                    ToTensor()])
	transforms = Compose(transforms_list, p=p)
	return lambda img:transforms(image=np.array(img))["image"]