import os
from skimage.io import imread
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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
