import torch
from torch import nn
from typing import Union
import torch_xla.core.xla_model as xm # <--- TPU Backend

from common import trunc_normal_init_

class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # TPU Note: Buffers are fine. xm.save will handle moving them to CPU for checkpoints.
        self.weights = nn.Buffer(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std), persistent=True
        )

        self.local_weights = nn.Buffer(torch.zeros(batch_size, embedding_dim, requires_grad=True), persistent=False)
        self.local_ids = nn.Buffer(torch.zeros(batch_size, dtype=torch.int32), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return self.weights[inputs].to(self.cast_to)
            
        with torch.no_grad():
            # TPU/XLA handles advanced indexing like weights[inputs] efficiently.
            self.local_weights.copy_(self.weights[inputs])
            self.local_ids.copy_(inputs)

        return self.local_weights.to(self.cast_to)


class CastedSparseEmbeddingSignSGD_Distributed(torch.optim.Optimizer):
    def __init__(self, params, world_size: int, lr: float = 1e-3, weight_decay: float = 1e-2):
        defaults = dict(lr=lr, weight_decay=weight_decay, world_size=world_size)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            local_weights_grad = None
            local_ids = None
            weights = None
            
            for p in group["params"]:
                if p.requires_grad:
                    local_weights_grad = p.grad
                elif p.ndim == 1:
                    local_ids = p
                elif p.ndim == 2:
                    weights = p

            if local_weights_grad is not None:
                _sparse_emb_signsgd_tpu(
                    local_weights_grad,
                    local_ids,
                    weights,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"]
                )

def _sparse_emb_signsgd_tpu(
    local_weights_grad: torch.Tensor,
    local_ids: torch.Tensor,
    weights: torch.Tensor,
    lr: float,
    weight_decay: float
) -> None:
    # 1. All-Gather using XLA
    # xm.all_gather is highly optimized on TPU v5e
    all_weights_grad = xm.all_gather(local_weights_grad)
    all_ids = xm.all_gather(local_ids)

    # 2. Handle Unique/Scatter
    # WARNING: unique() causes dynamic shapes. 
    # If vocab size (weights.shape[0]) is small, it's better to scatter into a fixed-size buffer.
    # If vocab size is massive, we use unique but expect a compilation hit on the first few steps.
    grad_ids, inv = torch.unique(all_ids, return_inverse=True)
    
    D = all_weights_grad.shape[1]
    grad = torch.zeros((grad_ids.shape[0], D), dtype=all_weights_grad.dtype, device=all_weights_grad.device)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, D), all_weights_grad)

    # 3. Apply Update
    p = weights[grad_ids]
    p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)

    # 4. Write back
    weights[grad_ids] = p