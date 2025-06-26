import torch
from components import MLP
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import zipfile
import os
import random
import shutil

class TransformerEncoderLayer(nn.Module):
  def __init__(self , embed_dim , num_heads , mlp_dim, drop_rate):
    super().__init__()
    self.norm1 = nn.LayerNorm(embed_dim)
    self.attn = nn.MultiheadAttention(embed_dim , num_heads , dropout = drop_rate , batch_first= True)
    self.norm2 = nn.LayerNorm(embed_dim)
    self.mlp = MLP(embed_dim , mlp_dim , drop_rate)
    self.drop = nn.Dropout(drop_rate)

  def forward(self, x):
    x_norm = self.norm1(x)
    x = x + self.attn(x_norm, x_norm, x_norm)[0]

    x_norm = self.norm2(x)
    x = x + self.mlp(x_norm)
    return x