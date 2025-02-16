from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import os
from typing import Optional, Callable

class BaseDataset(Dataset):
    def __init__(self, 
                 dataset: Dataset,
                 transform_fn: Optional[Callable] = None):
        self.dataset = dataset
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        if self.transform_fn is not None:
            image = self.transform_fn(image)
            
        return image, label

    def save_samples(self, experiment_name: str, num_samples: int = 10):
        """Save samples as a grid of images without requiring display"""
        os.makedirs('experiment_samples', exist_ok=True)
        
        clean_name = experiment_name.replace(" -> ", "_to_")
        clean_name = "".join(c for c in clean_name if c.isalnum() or c in (' ', '_', '-'))
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes_flat = axes.flatten()
        
        for i in range(num_samples):
            image, label = self[i]
            if isinstance(image, torch.Tensor):
                if image.dim() > 3:
                    image = image.squeeze(0)
                image = image.permute(1, 2, 0).numpy()
                image = np.clip(image, 0, 1)
            
            axes_flat[i].imshow(image)
            axes_flat[i].axis('off')
            axes_flat[i].set_title(f'Label: {label}')
        
        plt.suptitle(experiment_name, y=1.02, fontsize=12)
        plt.tight_layout()
        output_path = os.path.join('experiment_samples', f'{clean_name}.png')
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)