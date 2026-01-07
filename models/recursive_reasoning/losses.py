from typing import Any, Tuple, Dict, Sequence, Optional
import torch
import torch.nn.functional as F
from torch import nn

IGNORE_LABEL_ID = -100

def s(x, epsilon=1e-12):
    """
    A stable substitute for exp(x).
    TPU Note: Using 1e-12 instead of 1e-30 as 1e-30 can underflow in BF16/F32.
    """
    return torch.where(
        x < 0,
        1 / (1 - x + epsilon),
        x + 1
    )

def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))

def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    # TPU OPTIMIZATION: Use float32 instead of float64. 
    # XLA is significantly slower with double precision.
    logprobs = log_stablemax(logits.to(torch.float32), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    
    transformed_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))
    
    # gather logic is well-supported on TPU
    prediction_logprobs = torch.gather(
        logprobs, 
        index=transformed_labels.to(torch.long).unsqueeze(-1), 
        dim=-1
    ).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, torch.zeros_like(prediction_logprobs))

def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Standard cross_entropy in float32 is the most stable path for TPU
    logits_f32 = logits.to(torch.float32)
    labels_long = labels.to(torch.long)
    
    # Flattening for cross_entropy is a common TPU pattern
    loss = F.cross_entropy(
        logits_f32.reshape(-1, logits_f32.shape[-1]), 
        labels_long.reshape(-1), 
        ignore_index=ignore_index, 
        reduction="none"
    )
    return loss.reshape(labels.shape)

class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        # Dynamically select loss function
        if loss_type == "stablemax_cross_entropy":
            self.loss_fn = stablemax_cross_entropy
        else:
            self.loss_fn = softmax_cross_entropy
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        
        # 1. Forward through the reasoning model
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Calculate metrics (All metrics remain on device)
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            valid_metrics = new_carry.halted & (loss_counts > 0)
            
            # We perform summation here; global averaging happens in pretrain.py via xm.mesh_reduce
            metrics = {
                "count": valid_metrics.sum().to(torch.float32),
                "accuracy": torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), torch.zeros_like(loss_counts, dtype=torch.float32)).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum().to(torch.float32),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum().to(torch.float32),
                "steps": torch.where(valid_metrics, new_carry.steps.to(torch.float32), torch.zeros_like(new_carry.steps, dtype=torch.float32)).sum(),
            }

        # 2. Loss Calculation
        # Language Model Loss
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        
        # Halting Loss (Binary CE)
        # Ensure targets have the same dtype as logits for TPU efficiency
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"], 
            seq_is_correct.to(outputs["q_halt_logits"].dtype), 
            reduction="sum"
        )
        
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Bootstrapping target loss (Optional)
        q_continue_loss = torch.tensor(0.0, device=lm_loss.device, dtype=lm_loss.dtype)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"], 
                outputs["target_q_continue"], 
                reduction="sum"
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        # The total scalar loss
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)

        # halted.all() triggers a host-sync. This is necessary for the reasoning loop,
        # but XLA manages this internally.
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()