import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io

class LesionTrainDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0] + '.jpg')
        image = io.imread(img_name)
        label = int(self.data.iloc[idx, 1])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class LesionTestDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0] + '.jpg')
        image = io.imread(img_name)
        label = int(self.data.iloc[idx, 1])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Set paths
data_dir = 'path/to/dataset'  # Set your dataset path
train_csv = os.path.join(data_dir, 'train_labels.csv')
test_csv = os.path.join(data_dir, 'test_labels.csv')
train_img_dir = os.path.join(data_dir, 'train_images')
test_img_dir = os.path.join(data_dir, 'test_images')

# Transformations matching pretrained ResNet152
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#initializing datasets and loaders
train_dataset = LesionTrainDataset(train_csv, train_img_dir, transform=transform)
test_dataset = LesionTestDataset(test_csv, test_img_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Testing dataloaders
for imgs, labels in train_loader:
    print(imgs.shape, labels.shape)
    break
for imgs, labels in test_loader:
    print(imgs.shape, labels.shape)
    break


# Load Pretrained ResNet
model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)  # Updated for torchvision >= 0.13
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Binary classification (benign vs malignant)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
 
# Training loop
epochs = 30  # Adjusted to align with ResNet152 fine-tuning practices
train_losses = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

correct = 0
total = 0

#evaluating model and dont compute gradients
with torch.nograd():
  for imgs, labels in test_loader:
    batch_size = imgs.shape[0]
    outputs = models.eval()(imgs.view(batch_size, -1))
    values, predicted = torch.max(outputs, dim=1)
    total += labels.shape[0]
    correct += int((predicted == labels).sum())
print("Accuracy: %f" % (correct / total))

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid()
plt.show()

# Save model
torch.save(model.state_dict(), 'isic_resnet152.pth')
