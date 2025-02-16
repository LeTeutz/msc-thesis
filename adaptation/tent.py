import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseAdapter
from .registry import register_adapter

@register_adapter("tent")
class TENT(BaseAdapter):
    """TENT adaptation method"""
    def __init__(self, model: nn.Module, device: torch.device, lr: float = 1e-3):
        super().__init__(model, device)
        self.model.eval()
        
        print("\nDEBUG: Model structure:")
        for name, module in model.named_modules():
            print(f"Layer: {name}, Type: {type(module)}")
        
        self.tent_params = []
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                print(f"\nDEBUG: Found BatchNorm layer: {name}")
                print(f"Parameters before enabling grad:")
                print(f"- Weight requires_grad: {m.weight.requires_grad}")
                print(f"- Bias requires_grad: {m.bias.requires_grad}")
                
                m.requires_grad_(True)
                self.tent_params.extend([m.weight, m.bias])
                
                print(f"Parameters after enabling grad:")
                print(f"- Weight requires_grad: {m.weight.requires_grad}")
                print(f"- Bias requires_grad: {m.bias.requires_grad}")
        
        print(f"\nDEBUG: Collected {len(self.tent_params)} parameters for optimization")
        if not self.tent_params:
            raise ValueError("No BatchNorm parameters found in the model")
            
        self.optimizer = torch.optim.Adam(self.tent_params, lr=lr)

    def reset(self):
        """Reset optimizer state"""
        self.optimizer = torch.optim.Adam(self.tent_params, lr=self.optimizer.param_groups[0]['lr'])
        
    def entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Entropy minimization loss"""
        p = F.softmax(logits, dim=1)
        return -(p * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    
    @torch.enable_grad()  
    def adapt_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt batch norm parameters and predict"""
        self.optimizer.zero_grad()
        
        outputs = self.model(x)
        
        loss = self.entropy_loss(outputs)
        loss.backward()
        self.optimizer.step()
        
        return outputs.detach()