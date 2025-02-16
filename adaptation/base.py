import torch
import torch.nn as nn
from typing import Optional

class BaseAdapter:
    """Base class for test-time adaptation methods"""
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        
    def adapt_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt model and return predictions"""
        print(f"Base adapt_and_predict called for {self.__class__.__name__}")
        raise NotImplementedError
        
    def reset(self):
        """Reset adaptation state if needed"""
        pass