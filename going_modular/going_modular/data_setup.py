"""
 Contains functionality for creating PyTorch 
 DataLoader's for image classification data.
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir:str,
    test_dir: str,
    transform:transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS):
    """
        Creates training and testing dataloaders

        Takes a training and testing directory path and creates PyTorch Datasets
        and PyTorch dataloaders

        Args:
            train_dir: path to traning directory
            test_dir: path to testing directory
            transform: torchvision transforms to perform on data
            batch_size: num samples per batch in each dataloaders
            num_works: int for number of workers per dataloaders

        Returns:
            a tuple of (train_dataloader, test_dataloader, class_names)
            where class_names is a lits of target classes
            Example usage:
                train_dataloader, test_dataloaders, class_names = create_dataloaders(
                    train_dir=path/to/train_dir,
                    test_dir=path/to/test_dir,
                    transform=some_transform,
                    batch_size=32,
                    num_works=4)
    """
    # Imagefolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # class names
    class_names = train_data.classes

    # turn images into DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names

