import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import zipfile
import os
import random
import shutil

batch_size = 16
epochs = 50
learning_rate = 1e-4
patch_size = 8
num_classes = 4
image_size = 64
channels = 3
embed_dim = 256
num_heads = 16
depth = 12
mlp_dim = 512
drop_rate = 0.1
class data():
    def __init__(self,filepath,destination_path, image_size = 224,batch_size = 64):
        self.zip_path = filepath
        self.image_size = image_size
        self.batch_size = batch_size
        self.extract_to = destination_path
    def get_data(self):
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_to)

        transforms = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5) , (0.5)),
            transforms.Resize((image_size , image_size))
            ]
        )

        train_dataset = datasets.ImageFolder(root=self.extract_to +'Training', transform=transforms)
        test_dataset = datasets.ImageFolder(root=self.extract_to + 'Testing', transform=transforms)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size

        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader , val_loader , test_loader

