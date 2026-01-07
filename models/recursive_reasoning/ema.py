import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import copy

class EMAHelper:
    def __init__(self, mu: float = 0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, model: nn.Module):
        # Store initial shadow weights on the same device as the model (TPU)
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()
        for name, buffer in model.named_buffers():
            self.shadow[name] = buffer.data.clone().detach()

    def update(self, model: nn.Module):
        # We wrap the update in no_grad for speed
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Optimized in-place update for XLA fusion
                    new_average = (1.0 - self.mu) * param.data + self.mu * self.shadow[name]
                    self.shadow[name].copy_(new_average)
            
            for name, buffer in model.named_buffers():
                new_average = (1.0 - self.mu) * buffer.data + self.mu * self.shadow[name]
                self.shadow[name].copy_(new_average)
        
        # Crucial for TPU: We don't mark_step() inside the loop, 
        # but the update() call itself is usually followed by xm.mark_step() in pretrain.py
    
    def ema_copy(self, model: nn.Module):
        """Returns a copy of the model with EMA weights applied."""
        model_copy = copy.deepcopy(model)
        with torch.no_grad():
            for name, param in model_copy.named_parameters():
                if name in self.shadow:
                    param.data.copy_(self.shadow[name])
            for name, buffer in model_copy.named_buffers():
                if name in self.shadow:
                    buffer.data.copy_(self.shadow[name])
        return model_copy