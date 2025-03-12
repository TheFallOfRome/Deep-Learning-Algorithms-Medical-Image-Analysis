import os
from skimage.io import imread
import pandas as pd
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt

class lesion_train_Dataset(Dataset):
    #initializing
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    #returning length of csv
    def __len__(self):
        return len(self.data)
    
    #returning images and csv labels
    def __getitem__(self, index):
        image_path = os.path.join(self.image_folder, self.data.iloc[index, 0]) 
        image = imread(image_path)
        y_label = torch.tensor(int(self.data.iloc[index, 1], dtype=torch.long))

        if self.transform:
            image = self.transform(image)
        
        return (image, y_label)


class lesion_test_Dataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_folder, self.data.iloc[index, 0])
        image = imread(image_path)
        y_label = torch.tensor(int(self.data.iloc[index, 1]), dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

#transformations inline with pretrained resnet152
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = lesion_train_Dataset(
    csv_file='ISBI2016_ISIC_Part3_Training_GroundTruth.csv',
    image_folder='ISBI2016_ISIC_Part3_Training_Data',
    transform=transforms.ToTensor()
)

test_dataset = lesion_test_Dataset(
    csv_file='ISBI2016_ISIC_Part3_Test_GroundTruth.csv',
    image_folder='ISBI2016_ISIC_Part3_Test_Data',
    transform=transforms.ToTensor()
)


train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

#testing dataloaders
for imgs, labels in train_dataloader:
    print(imgs.shape, labels.shape)
    break
for imgs, labels in test_dataloader:
    print(imgs.shape, labels.shape)
    break

#loading pretrained resnet152
model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2) 

#changing last layer to binary classification
num_features = model.fc.in_features #num of input features for last layer
model.fc = nn.Linear(num_features, 2)  #benign vs malignant

#use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
train_losses = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_dataloader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

#plotting training loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid()
plt.show()