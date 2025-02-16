import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import BaseAdapter
from .registry import register_adapter


@register_adapter("t3a")
class T3A(BaseAdapter):
    def __init__(self, model: nn.Module, device: torch.device, num_classes: int = 10,
                 momentum: float = 0.9, base_temp: float = 0.1):
        super().__init__(model, device)
        self.model.eval()
        self.momentum = momentum
        
        
        if hasattr(model, 'fc_layers'):
            for module in reversed(model.fc_layers):
                if isinstance(module, nn.Linear):
                    self.num_classes = module.out_features
                    break
        else:
            self.num_classes = num_classes
            
        print(f"DEBUG T3A: Initializing with {self.num_classes} classes")
        
        self.base_temp = base_temp
        self.temp = self._init_temperature(self.num_classes)
        
        
        self.prototypes = {c: None for c in range(self.num_classes)}
        self.features = []
        
        penultimate = None
        last_linear = None
        for module in model.fc_layers:
            if isinstance(module, nn.Linear):
                if last_linear is not None:
                    penultimate = last_linear
                last_linear = module
        
        if penultimate is None:
            raise ValueError("Could not find penultimate linear layer")
        
        def hook_fn(module, input, output):
            self.features = output.detach()
            
        penultimate.register_forward_hook(hook_fn)
        print(f"DEBUG T3A: Registered hook on layer with output size {penultimate.out_features}")


    def _init_temperature(self, num_classes: int) -> float:
        return self.base_temp * (1 + torch.log(torch.tensor(num_classes / 10.0))).item()

    def update_prototypes(self, features: torch.Tensor, outputs: torch.Tensor):
        pseudo_labels = outputs.softmax(dim=1).argmax(dim=1)
        
        for c in range(self.num_classes):
            mask = (pseudo_labels == c)
            if mask.sum() > 0:
                class_features = features[mask].mean(0, keepdim=True)
                if self.prototypes[c] is None:
                    self.prototypes[c] = class_features
                else:
                    self.prototypes[c] = (
                        self.momentum * self.prototypes[c] + 
                        (1 - self.momentum) * class_features
                    )

    def get_similarities(self, features: torch.Tensor) -> torch.Tensor:
        """Compute similarities with dimensionality check"""
        similarities = []
        features = F.normalize(features, dim=1)
        
        for c in range(self.num_classes):
            if self.prototypes[c] is not None:
                prototype = F.normalize(self.prototypes[c], dim=1)
                sim = (features @ prototype.t()) / self.temp
                similarities.append(sim)
            else:
                similarities.append(torch.zeros(features.size(0), 1).to(self.device))
        
        similarities = torch.cat(similarities, dim=1)
        return similarities

    def adapt_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt prototypes and return adjusted predictions with dimension checking"""
        outputs = self.model(x)  
        
        if len(self.features) == 0:
            return outputs
            
        self.update_prototypes(self.features, outputs)
        
        similarities = self.get_similarities(self.features)  
        
        if similarities.size(1) != outputs.size(1):
            raise ValueError(f"Dimension mismatch: similarities {similarities.size()}, outputs {outputs.size()}")
            
        adjusted_outputs = outputs + similarities
        
        return adjusted_outputs

    def reset(self):
        self.prototypes = {c: None for c in range(self.num_classes)}
        self.temp = self._init_temperature(self.num_classes)