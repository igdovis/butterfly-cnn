from torch.utils.data import Dataset
import os 
import pandas as pd
from torchvision.io import read_image

class ButterflyDataset(Dataset):
    def __init__(self, labels, img_dir, transform=None, target_transform=None):
        self.image_labels = pd.read_csv(labels)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.image_labels.iloc[index, 0])
        image = read_image(img_path)
        label = self.image_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
def fun():
    print(5)

    


