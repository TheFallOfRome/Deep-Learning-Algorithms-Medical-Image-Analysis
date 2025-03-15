import os
from skimage import io, color
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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