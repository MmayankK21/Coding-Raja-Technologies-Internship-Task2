import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.utils import download_and_extract_archive


def download_food101(data_dir):
    url = 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'
    download_path = os.path.join(data_dir, 'food-101.tar.gz')
    extract_path = os.path.join(data_dir, 'food-101')

    if not os.path.exists(extract_path):
        print("Downloading and extracting Food-101 dataset...")
        download_and_extract_archive(url, download_root=data_dir)
    else:
        print("Food-101 dataset already downloaded.")


def get_dataloaders(data_dir, batch_size=32, split_ratio=0.8):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader