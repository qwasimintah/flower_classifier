# Imports here
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models



train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])



def data_loaders(dir, transforms, batch_size):
	# TODO: Load the datasets with ImageFolder
	image_datasets = datasets.ImageFolder(dir, transform=transforms)

	# TODO: Using the image datasets and the trainforms, define the dataloaders
	dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)


	return dataloaders




