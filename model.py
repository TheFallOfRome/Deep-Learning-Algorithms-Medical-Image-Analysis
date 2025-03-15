import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn

#loading pretrained resnet
model = models.resnet152(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  #benign vs malignant


#use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Loss and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
