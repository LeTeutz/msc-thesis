from torchvision import datasets, transforms
import torch
from typing import Optional, Callable
from .base import BaseDataset
import random
import numpy as np

# mnist_reduced_100.py
class MNISTReduced100(BaseDataset):
    def __init__(self,
                 root: str = "./data",
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 download: bool = True):
        
        full_dataset = datasets.MNIST(
            root=root,
            train=train,
            transform=None,  
            download=download
        )
        
        random.seed(2025)
        np.random.seed(2025)
        
        indices = []
        for class_idx in range(10):
            class_indices = [i for i, (_, label) in enumerate(full_dataset) if label == class_idx]
            selected_indices = random.sample(class_indices, 100)
            indices.extend(selected_indices)
        
        reduced_dataset = torch.utils.data.Subset(full_dataset, indices)
        
        super().__init__(reduced_dataset, transform_fn=transform)

    @property
    def classes(self):
        return list(range(10))