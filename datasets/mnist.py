from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch
from typing import Optional, Callable
from .base import BaseDataset

class MNISTDataset(BaseDataset):
    def __init__(self, 
                 root: str = "./data",
                 train: bool = True,
                 transform_fn: Optional[Callable] = None,
                 download: bool = True):

        dataset = datasets.MNIST(
            root=root,
            train=train,
            transform=transforms.ToTensor(),
            download=download
        )
        super().__init__(dataset, transform_fn)

    @property
    def classes(self):
        return list(range(10))