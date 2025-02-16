import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseAdapter
from .registry import register_adapter

@register_adapter("style")
class StyleTransfer(BaseAdapter):
   """Test-Time Style Shifting adapts samples by matching feature statistics
   to the nearest source distribution style through feature transformation."""
   
   def __init__(self, 
                model: nn.Module, 
                device: torch.device,
                style_momentum: float = 0.9,
                num_styles: int = 16,
                style_strength: float = 0.5,
                debug: bool = True):
       super().__init__(model, device)
       self.model.eval()
       self.momentum = style_momentum
       self.num_styles = num_styles
       self.style_strength = style_strength
       self.debug = debug
       
       self.features = {}
       self.adapted_features = {}
       self.style_bank = []
       self.hooks = []
       self._register_hooks()
       
   def _register_hooks(self):
       """Register hooks for feature extraction"""
       def get_hook(name):
           def hook(module, input, output):
               if name in self.adapted_features:
                   return self.adapted_features[name]
               self.features[name] = output
           return hook
       
       for name, module in self.model.named_modules():
           if isinstance(module, nn.Conv2d):
               hook = module.register_forward_hook(get_hook(name))
               self.hooks.append(hook)
       
       if self.debug:
           print(f"Registered {len(self.hooks)} hooks on conv layers")
   
   def compute_statistics(self, feat: torch.Tensor) -> tuple:
       """Compute channel-wise mean and std, averaging over batch"""
       b, c = feat.size(0), feat.size(1)
       feat_flat = feat.view(b, c, -1)
       feat_flat = feat_flat.mean(dim=0, keepdim=True)  
       mean = feat_flat.mean(dim=2).view(1, c, 1, 1)
       std = feat_flat.std(dim=2).view(1, c, 1, 1) + 1e-8
       return mean, std
   
   def get_style_distance(self, content_stats: tuple, style_stats: tuple) -> torch.Tensor:
       """Compute distance between feature statistics"""
       c_mean, c_std = content_stats
       s_mean, s_std = style_stats
       
       mean_dist = torch.mean((c_mean - s_mean) ** 2)
       std_dist = torch.mean((c_std - s_std) ** 2)
       
       return mean_dist + std_dist
   
   def update_style_bank(self, features: dict):
       """Update style bank with new statistics"""
       new_style = {}
       for name, feat in features.items():
           new_style[name] = self.compute_statistics(feat)
       
       if len(self.style_bank) < self.num_styles:
           self.style_bank.append(new_style)
           if self.debug:
               print(f"Added new style. Bank size: {len(self.style_bank)}")
       else:
           min_dist = float('inf')
           min_idx = 0
           
           for i, style in enumerate(self.style_bank):
               dist = sum(self.get_style_distance(style[name], new_style[name]) 
                        for name in new_style.keys())
               if dist < min_dist:
                   min_dist = dist
                   min_idx = i
           
           updated_style = {}
           for name in new_style.keys():
               c_mean, c_std = self.style_bank[min_idx][name]
               n_mean, n_std = new_style[name]
               
               updated_style[name] = (
                   self.momentum * c_mean + (1 - self.momentum) * n_mean,
                   self.momentum * c_std + (1 - self.momentum) * n_std
               )
           
           self.style_bank[min_idx] = updated_style
           if self.debug:
               print(f"Updated style {min_idx} with distance {min_dist:.4f}")
   
   def adapt_features(self, content_feat: torch.Tensor, 
                     content_stats: tuple, style_stats: tuple) -> torch.Tensor:
       """Transform features to match style statistics"""
       c_mean, c_std = content_stats
       s_mean, s_std = style_stats
       
       b = content_feat.size(0)
       s_mean = s_mean.expand(b, -1, 1, 1)
       s_std = s_std.expand(b, -1, 1, 1)
       c_mean = c_mean.expand(b, -1, 1, 1)
       c_std = c_std.expand(b, -1, 1, 1)
       
       normalized = (content_feat - c_mean) / c_std
       styled = normalized * s_std + s_mean
       
       return self.style_strength * styled + (1 - self.style_strength) * content_feat
   
   def adapt_and_predict(self, x: torch.Tensor) -> torch.Tensor:
       """Adapt features and make prediction"""
       self.features.clear()
       self.adapted_features.clear()
       
       with torch.no_grad():
           original_output = self.model(x)
           if not self.features:
               return original_output
           original_features = {k: v.detach() for k, v in self.features.items()}
           
       self.update_style_bank(original_features)
       
       if not self.style_bank:
           return original_output
           
       current_stats = {name: self.compute_statistics(feat) 
                       for name, feat in original_features.items()}
       
       min_dist = float('inf')
       best_style = None
       
       for style in self.style_bank:
           dist = sum(self.get_style_distance(current_stats[name], style[name]) 
                     for name in style.keys())
           if dist < min_dist:
               min_dist = dist
               best_style = style
       
       if self.debug:
           print(f"Best style distance: {min_dist:.4f}")
               
       for name, feat in original_features.items():
           self.adapted_features[name] = self.adapt_features(
               feat, current_stats[name], best_style[name]
           )
           
       with torch.no_grad():
           adapted_output = self.model(x)
           
           if self.debug and torch.allclose(original_output, adapted_output):
               print("Warning: Adaptation had no effect on output")
           
           return adapted_output
           
   def reset(self):
       """Reset adaptation state"""
       self.features.clear()
       self.adapted_features.clear()
       self.style_bank = []