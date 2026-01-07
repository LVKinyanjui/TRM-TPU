from typing import Tuple, Optional
import einops
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

from .common import trunc_normal_init_

CosSin = Tuple[torch.Tensor, torch.Tensor]

def _find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b

def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    # Using reshape instead of view for stability with non-contiguous tensors
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    # RoPE is sensitive to precision; we compute in the higher precision of the table (usually float32)
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    # Unsqueeze to align with [bs, seq_len, num_heads, head_dim]
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = nn.Parameter(torch.zeros((out_features,))) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Explicit casting allows XLA to fuse the 'cast-and-multiply' into a single TPU operation
        return F.linear(input, self.weight.to(input.dtype), 
                        bias=self.bias.to(input.dtype) if self.bias is not None else None)

class CastedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float, device=None):
        super().__init__()
        # 1. Compute on the target device immediately
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # 2. Register as buffers so they move with the model and are saved in checkpoints
        # persistent=False means they aren't stored in state_dict (since they are generated)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self) -> CosSin:
        return self.cos_cached, self.sin_cached

class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)

        # Use reshape instead of view to prevent "non-contiguous" errors 
        # (Safer for XLA graph tracing)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Rearrange for Scaled Dot Product Attention (B, H, S, D)
        query, key, value = map(lambda t: t.transpose(1, 2), (query, key, value))
        
        # TPU/XLA automatically optimizes this call to use high-perf hardware kernels
        attn_output = scaled_dot_product_attention(
            query=query, 
            key=key, 
            value=value, 
            is_causal=self.causal
        )
        
        # Reshape back to (B, S, OutputSize)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        # Using chunk and SiLU is a standard pattern that XLA fuses well
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    # TPU v5e handles BF16 well, but we use float32 for the mean reduction to maintain stability
    hidden_states_fp32 = hidden_states.to(torch.float32)
    variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
    hidden_states_fp32 = hidden_states_fp32 * torch.rsqrt(variance + variance_epsilon)
    return hidden_states_fp32.to(input_dtype)