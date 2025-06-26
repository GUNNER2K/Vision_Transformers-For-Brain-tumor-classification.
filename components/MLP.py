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

class MLP(nn.Module):
  def __init__(self , input_features , hidden_size = 1024 , dropout_rate = 0.1):
    super().__init__()
    self.fc1 = nn.Linear(input_features , hidden_size)
    self.fc2 = nn.Linear(hidden_size , input_features)
    self.dropout = nn.Dropout(dropout_rate)


  def forward(self , x):
    x = self.fc1(x)
    x = F.gelu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.dropout(x)
    return x