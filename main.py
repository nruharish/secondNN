import torch
import torch.nn as nn
from Model import Model
from torchinfo import summary


#create a dataset
features = torch.rand(10,5)

model = Model(features.shape[1])
out = model(features)

summary(model, input_size=(10,5))