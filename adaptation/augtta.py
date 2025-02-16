import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from .base import BaseAdapter
from .registry import register_adapter

@register_adapter("augtta")
class AugTTA(BaseAdapter):
    """Test-time Adaptation using Augmentations"""
    def __init__(self, model: nn.Module, device: torch.device, num_augments: int = 32):
        super().__init__(model, device)
        self.model.eval()
        self.num_augments = num_augments
        
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        ])
        
    def adapt_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict using multiple augmented versions"""
        all_outputs = []
        orig_output = self.model(x)
        all_outputs.append(orig_output)
        
        for _ in range(self.num_augments):
            aug_x = self.augment(x)
            aug_output = self.model(aug_x)
            all_outputs.append(aug_output)

        outputs = torch.stack(all_outputs, dim=0)
        return outputs.mean(0)