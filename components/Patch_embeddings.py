
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
import zipfile
import os
import random
import shutil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Entered patch embedding, using device :" ,device)

class PatchEmbedding(nn.Module):
  def __init__(self , img_size = 224 , patch_size = 16, in_channels = 3 , embed_dim = 768):
    super().__init__()
    self.path_size = patch_size
    self.proj = nn.Conv2d(in_channels , embed_dim , kernel_size = patch_size , stride = patch_size)
    num_patches = (img_size // patch_size) ** 2
    self.cls_token = nn.Parameter(torch.randn(1 , 1 , embed_dim))
    self.pos_embedding = nn.Parameter(torch.randn(1 , num_patches + 1 , embed_dim))

  def forward(self , x):
    B = x.size(0)
    x = self.proj(x) # (B , E , H/P , W/P)
    x = x.flatten(2).transpose(1 , 2) # ( B , N, E)
    cls_token = self.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_token , x), dim = 1)
    x = x + self.pos_embedding
    return x