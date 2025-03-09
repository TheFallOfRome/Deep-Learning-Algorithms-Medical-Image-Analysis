import torch
import torchvision.models as models
import torch.nn as nn

#loading pretrained resnet152
model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2) 

#changing last layer to binary classification
num_features = model.fc.in_features #num of input features for last layer
model.fc = nn.Linear(num_features, 2)  #benign vs malignant

#use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
