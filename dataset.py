import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SportBallsDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and labels.
            train (bool): Flag to indicate if the dataset is for training or testing.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        
        if self.train:
            self.labels_file = os.path.join(root_dir, 'Train', 'labels.csv')
            self.images_dir = os.path.join(root_dir, 'Train')
            self.labels_df = pd.read_csv(self.labels_file, header=None)
        else:
            self.labels_file = os.path.join(root_dir, 'Test', 'labels.csv')
            self.images_dir = os.path.join(root_dir, 'Test')
            self.labels_df = pd.read_csv(self.labels_file, header=None)


    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_dir, f"{self.labels_df.iloc[idx, 0]}.png")
        label = self.labels_df.iloc[idx, 1].astype(int)

        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


if __name__=="__main__":
    # Define transforms (optional)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create full training dataset
    full_train_dataset = SportBallsDataset(root_dir='dataset', train=True, transform=transform)
    # Create test dataset
    test_dataset = SportBallsDataset(root_dir='dataset', train=False, transform=transform)

    print(len(full_train_dataset))
    print(len(test_dataset))

    sample = full_train_dataset[0]
    print(sample[1])
    plt.imshow(sample[0].permute(1,2,0).numpy())
    plt.show()

    sample = test_dataset[2]
    print(sample[1])
    plt.imshow(sample[0].permute(1,2,0).numpy())
    plt.show()