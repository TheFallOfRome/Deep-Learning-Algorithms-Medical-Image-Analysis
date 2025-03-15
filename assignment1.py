import os
from skimage import io, color
import pandas as pd
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt

class lesion_train_Dataset(Dataset):
    def __init__(self, csv_file, image_folder, transforms):
        self.transform = transforms
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        

    #returning length of csv
    def __len__(self):
        return len(self.data)
    
    #returning images and csv labels
    def __getitem__(self, index):
        image_name = self.data.iloc[index, 0]
        label = self.data.iloc[index, 1]
        image_path = os.path.join(self.image_folder, image_name) 
        image = io.imread(image_path)

        #change grayscale to rgb for resnet
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[-1] != 3):
            image = color.gray2rgb(image)

        if self.transform:
            image = self.transform(image)
        
        return (image, label)


class lesion_test_Dataset(Dataset):
    def __init__(self, csv_file, image_folder, transforms):
        self.transform = transforms
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        

    #returning length of csv
    def __len__(self):
        return len(self.data)
    
    #returning images and csv labels
    def __getitem__(self, index):
        image_name = self.data.iloc[index, 0]
        label = self.data.iloc[index, 1]
        image_path = os.path.join(self.image_folder, image_name) 
        image = io.imread(image_path)

        #change grayscale to rgb for resnet
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[-1] != 3):
            image = color.gray2rgb(image)

        if self.transform:
            image = self.transform(image)
        
        return (image, label)

#transformations inline with pretrained resnet152
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#loading training and testing datasets
train_dataset = lesion_train_Dataset(
    csv_file='/home/juancervantes02/Desktop/4341/ISBI2016_ISIC_Part3_Training_GroundTruth.csv',
    image_folder='/home/juancervantes02/Desktop/4341/ISBI2016_ISIC_Part3_Training_Data',
    transforms=transform
)

test_dataset = lesion_test_Dataset(
    csv_file='/home/juancervantes02/Desktop/4341/ISBI2016_ISIC_Part3_Test_GroundTruth.csv',
    image_folder='/home/juancervantes02/Desktop/4341/ISBI2016_ISIC_Part3_Test_Data',
    transforms=transform
)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

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

epochs = 25
train_losses = []  #for plotting loss

#training the model
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
    
    #average loss per epoch
    epoch_loss = running_loss / len(train_dataloader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}\n")

#testing accuracy on test set

with torch.no_grad():
    correct = 0
    total = 0
    model.eval()
    #iterating through test set
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.4f}%\n')

#creating training loss figure
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid()
plt.show()

