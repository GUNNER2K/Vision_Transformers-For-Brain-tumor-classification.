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
from components import VisionTransformer
from components import Data_Ingestion

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
source = 
destination = 
data = Data_Ingestion()
model = VisionTransformer(image_size ,
                          patch_size ,
                          channels ,
                          num_classes ,
                          embed_dim ,
                          depth ,
                          num_heads ,
                          mlp_dim ,
                          drop_rate).to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters() , lr = learning_rate)

def train(model , loader , optimizer , loss_fn):
  print('Entered Training')
  model.train()

  total_loss , correct = 0 , 0

  for x , y in loader:
    # print(y.unique())
    x , y = x.to(device) , y.to(device)
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred , y)
    loss.backward()
    optimizer.step()

    total_loss = total_loss + loss.item() * x.size(0)
    correct = correct + (pred.argmax(1) == y).sum().item()

  total_loss = total_loss / len(loader.dataset)
  correct = correct / len(loader.dataset)

  return total_loss , correct

def evaluate(model , loader , loss_fn):

  model.eval()
  correct = 0
  total_loss = 0

  with torch.inference_mode():
    for x , y in loader:
      x,y = x.to(device) , y.to(device)
      pred = model(x)
      loss = loss_fn(pred , y)
      total_loss = total_loss + loss.item() * x.size(0)
      correct = correct + (pred.argmax(1) == y).sum().item()

    correct = correct / len(loader.dataset)
    total_loss = total_loss / len(loader.dataset)
    return total_loss , correct
  
train_acc_list, test_acc_list = [], []
train_loss_list, test_loss_list = [], []
for epoch in range(epochs):
  print(f'Epoch: {epoch +1}/{epochs}')
  train_loss , train_acc = train(model , train_loader , optimizer , loss)
  test_loss , test_acc = evaluate(model , test_loader, loss)
  train_acc_list.append(train_acc)
  test_acc_list.append(test_acc)
  train_loss_list.append(train_loss)
  test_loss_list.append(test_loss)
  print(f'loss: {train_loss}  accuracy: {train_acc:.4f} val_loss: {test_loss}Validation accuracy: {test_acc:.4f}')


plt.plot(train_acc_list, label="Train Accuracy")
plt.plot(test_acc_list, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Test Accuracy")
plt.show()

plt.plot(train_loss_list, label="Train loss")
plt.plot(test_loss_list, label="val loss")
plt.xlabel("loss")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Test loss")
plt.show()