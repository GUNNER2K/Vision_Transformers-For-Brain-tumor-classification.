import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
import zipfile
import os
import random
import shutil
from components import Patch_embeddings
from components import Encoder

class VisionTransformer(nn.Module):
  def __init__(self , img_size , patch_size , input_channels , num_classes , embed_dim , depth , num_heads , mlp_dim , drop_rate):
    super().__init__()
    self.patch_embed = Patch_embeddings(img_size , patch_size , input_channels , embed_dim)
    self.encoder  = nn.Sequential(
       *[Encoder(embed_dim , num_heads , mlp_dim , drop_rate) for _ in range(depth)]
    )
    self.norm = nn.LayerNorm(embed_dim)
    self.FC = nn.Linear(embed_dim , num_classes)

  def forward(self , x):
    x = self.patch_embed(x)
    x = self.encoder(x)
    x = self.norm(x)
    cls_token = x[: , 0]
    x = self.FC(cls_token)
    return x