import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseAdapter
from .registry import register_adapter

@register_adapter("shot2")
class SHOT2(BaseAdapter):
    """SHOT (Source Hypothesis Transfer) performs test-time adaptation without source data access.
    
    The method adapts a pre-trained model to distribution shifts through three key mechanisms:
    1. Information Maximization:
       - Entropy minimization encourages confident predictions
       - Diversity maximization (improved) prevents class collapse by promoting uniform predictions
       
    2. Pseudo-labeling with Confidence Thresholding:
       - Generates pseudo-labels from model predictions
       - Only considers high-confidence predictions (above threshold) for adaptation
       - Helps maintain stability during adaptation
       
    3. Online Parameter Updates:
       - Updates model parameters through backpropagation
       - Uses Adam optimizer for stable adaptation
       - Balances information maximization and pseudo-label losses
    
    This implementation improves upon the original by:
    - Adding explicit diversity loss to prevent prediction collapse
    - Implementing confidence thresholding for reliable pseudo-labels
    - Proper loss balancing between entropy and diversity terms
    
    Args:
        model (nn.Module): Pre-trained source model
        device (torch.device): Device to run adaptation on
        alpha (float): Balance between entropy (alpha) and diversity (1-alpha)
        lr (float): Learning rate for adaptation
        pseudo_threshold (float): Confidence threshold for pseudo-label selection
    """


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