import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseAdapter
from .registry import register_adapter

@register_adapter("shot")
class SHOT(BaseAdapter):
    """Source Hypothesis Transfer with minimal entropy"""
    def __init__(self, model: nn.Module, device: torch.device, 
                 alpha: float = 0.9, lr: float = 1e-3):
        super().__init__(model, device)
        self.model.eval()
        self.alpha = alpha
        
        for param in self.model.parameters():
            param.requires_grad_(True)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
    def im_loss(self, outputs):
        """Information maximization loss"""
        softmax_out = F.softmax(outputs, dim=1)
        entropy_loss = torch.mean(torch.sum(-softmax_out * 
                                          F.log_softmax(outputs, dim=1), dim=1))
        msoftmax = softmax_out.mean(dim=0)
        diversity_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
        return entropy_loss - diversity_loss
    
    @torch.enable_grad()
    def adapt_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt using information maximization and predict"""
        self.optimizer.zero_grad()
        
        outputs = self.model(x)
        loss = self.im_loss(outputs)
        
        loss.backward()
        self.optimizer.step()
        
        return outputs.detach()

    def reset(self):
        """Reset optimizer state"""
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                        lr=self.optimizer.param_groups[0]['lr'])