from torchvision import datasets, transforms
import torch
from typing import Optional, Callable
from .base import BaseDataset

class MNIST1D(BaseDataset):
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
        
        class FlattenedDataset:
            def __init__(self, original_dataset):
                self.dataset = original_dataset
                
            def __len__(self):
                return len(self.dataset)
                
            def __getitem__(self, idx):
                image, label = self.dataset[idx]
                flattened = image.view(-1)
                flattened = flattened.unsqueeze(0)
                return flattened, label
                
        flattened_dataset = FlattenedDataset(dataset)
        super().__init__(flattened_dataset, transform_fn)
        
    def save_samples(self, experiment_name: str, num_samples: int = 10):
        """Override save_samples to reshape 1D back to 2D for visualization"""
        import matplotlib.pyplot as plt
        import os
        import numpy as np
        
        os.makedirs('experiment_samples', exist_ok=True)
        
        clean_name = experiment_name.replace(" -> ", "_to_")
        clean_name = "".join(c for c in clean_name if c.isalnum() or c in (' ', '_', '-'))
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes_flat = axes.flatten()
        
        for i in range(num_samples):
            image, label = self[i]
            if isinstance(image, torch.Tensor):
                image = image.view(28, 28)
                if image.dim() > 2:
                    image = image.squeeze(0)
                image = image.numpy()
                image = np.clip(image, 0, 1)
            
            axes_flat[i].imshow(image, cmap='gray')
            axes_flat[i].axis('off')
            axes_flat[i].set_title(f'Label: {label}')
        
        plt.suptitle(experiment_name, y=1.02, fontsize=12)
        plt.tight_layout()
        output_path = os.path.join('experiment_samples', f'{clean_name}.png')
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    @property
    def classes(self):
        return list(range(10))