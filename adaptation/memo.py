import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from .base import BaseAdapter
from .registry import register_adapter

@register_adapter("memo")
class MEMO(BaseAdapter):
    class RandomHFlip(nn.Module):
        """Random horizontal flip module"""
        def __init__(self, p=0.5):
            super().__init__()
            self.p = p
            
        def forward(self, x):
            if torch.rand(1) < self.p:
                return x.flip(-1)
            return x
            
    class RandomCrop(nn.Module):
        """Random crop and resize module"""
        def __init__(self, size=28, padding=4):
            super().__init__()
            self.size = size
            self.padding = padding
            
        def forward(self, x):
            padded = F.pad(x, [self.padding] * 4, mode='reflect')
            _, _, h, w = padded.shape
            
            y = torch.randint(0, h - self.size + 1, (1,))
            x = torch.randint(0, w - self.size + 1, (1,))
            
            return padded[:, :, y:y+self.size, x:x+self.size]
            
    class ColorJitter(nn.Module):
        """Color jitter module"""
        def __init__(self, brightness=0.2):
            super().__init__()
            self.brightness = brightness
            
        def forward(self, x):
            factor = 1.0 + (torch.rand(1) * 2 - 1) * self.brightness
            return torch.clamp(x * factor, 0, 1)

    def __init__(self, model: nn.Module, device: torch.device, 
                 lr: float = 1e-4,  
                 n_augmentations: int = 4,
                 temperature: float = 2.0,  
                 memo_alpha: float = 0.3):  
        super().__init__(model, device)
        self.model.eval()
        self.n_augmentations = n_augmentations
        self.temperature = temperature
        self.memo_alpha = memo_alpha
        
        for param in self.model.parameters():
            param.requires_grad_(True)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.augment = nn.Sequential(
            self.RandomHFlip(),
            self.RandomCrop(padding=2),
            self.ColorJitter(brightness=0.1) 
        )

    def entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy loss with improved numerical stability"""
        eps = 1e-8
        
        logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        logits_temp = (logits - logits_max.detach()) / self.temperature
        probs = F.softmax(logits_temp, dim=-1)
        
        avg_probs = probs.mean(dim=0)
        
        ent = -(avg_probs * torch.log(avg_probs.clamp(min=eps))).sum(dim=-1).mean()
        return ent

    def consistency_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute consistency loss with improved numerical stability"""
        logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        logits_temp = (logits - logits_max.detach()) / self.temperature
        probs = F.softmax(logits_temp, dim=-1)
        
        n_views = probs.size(0)
        total_loss = 0.0
        
        for i in range(n_views):
            for j in range(i + 1, n_views):
                p_i = probs[i].clamp(min=1e-8)
                p_j = probs[j].clamp(min=1e-8)
                
                loss = 0.5 * (
                    (p_i * (torch.log(p_i) - torch.log(p_j))).sum(dim=-1).mean() +
                    (p_j * (torch.log(p_j) - torch.log(p_i))).sum(dim=-1).mean()
                )
                total_loss += loss
        
        return total_loss / (n_views * (n_views - 1) / 2)

    @torch.enable_grad()
    def adapt_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt using augmented views"""
        self.optimizer.zero_grad()
        
        views = [x]  
        for _ in range(self.n_augmentations - 1):
            aug_x = x.clone()
            aug_x = self.augment(aug_x)
            views.append(aug_x)
        
        all_logits = []
        for view in views:
            logits = self.model(view)
            all_logits.append(logits)
        
        all_logits = torch.stack(all_logits)
        
        ent_loss = self.entropy_loss(all_logits)
        cons_loss = self.consistency_loss(all_logits)
        total_loss = self.memo_alpha * ent_loss + (1 - self.memo_alpha) * cons_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        with torch.no_grad():
            return self.model(x)