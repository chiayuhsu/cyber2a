import torch

from torchvision import models
from torchvision.models.resnet import ResNet18_Weights

# Load the pretrained ResNet18 model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)