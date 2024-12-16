from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Callable
from .base import BaseDataset


class DatasetLoader:
    def __init__(self, seed: int = 2025):
        self.seed = seed
        
    def extend_to_rgb(self, x):
        """Convert single channel image to 3 channels by repeating"""
        # TIL: black and white images have all 3 channels the same
        if x.size(0) == 1:  
            return x.repeat(3, 1, 1)
        return x
        
    def load_dataset(self, 
                    dataset_name: str,
                    train_transform: Optional[Callable] = None,
                    test_transform: Optional[Callable] = None) -> Tuple[DataLoader, DataLoader]:
        
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            self.extend_to_rgb
        ])
        
        if dataset_name == "mnist":
            train_full = datasets.MNIST(root="./data", train=True, download=True, transform=base_transform)
            test_full = datasets.MNIST(root="./data", train=False, download=True, transform=base_transform)
        elif dataset_name == "fashion_mnist":
            train_full = datasets.FashionMNIST(root="./data", train=True, download=True, transform=base_transform)
            test_full = datasets.FashionMNIST(root="./data", train=False, download=True, transform=base_transform)
        elif dataset_name == "emnist":
            train_full = datasets.EMNIST(root="./data", split='balanced', train=True, download=True, transform=base_transform)
            test_full = datasets.EMNIST(root="./data", split='balanced', train=False, download=True, transform=base_transform)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Wrap datasets with additional transformations if provided
        train_dataset = BaseDataset(train_full, transform_fn=train_transform)
        test_dataset = BaseDataset(test_full, transform_fn=test_transform)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        return train_loader, test_loader