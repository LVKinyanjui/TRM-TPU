from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from ..common import trunc_normal_init_
from ..layers import (
    rms_norm, SwiGLU, Attention, RotaryEmbedding, 
    CosSin, CastedEmbedding, CastedLinear
)
from ..sparse_embedding import CastedSparseEmbedding

# TPU/XLA specific: use xm to get the current device safely if needed
import torch_xla.core.xla_model as xm

IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor

@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]

class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int
    H_cycles: int
    L_cycles: int
    L_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int
    halt_exploration_prob: float
    forward_dtype: str = "bfloat16"
    mlp_t: bool = False 
    puzzle_emb_len: int = 16 
    no_ACT_continue: bool = True 

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        if self.config.mlp_t:
            # TPU Optimization: Ensure dimensions are consistent for XLA fusion
            self.puzzle_emb_len = (-(self.config.puzzle_emb_ndim // -self.config.hidden_size) 
                                   if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len)
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len,
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            # Transpose is efficient on TPU; reshape is used for safeguard
            h = hidden_states.transpose(1, 2)
            out = self.mlp_t(h)
            hidden_states = rms_norm(h + out, variance_epsilon=self.norm_eps).transpose(1, 2)
        else:
            # Residual + Norm + Attention
            attn_out = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
            hidden_states = rms_norm(hidden_states + attn_out, variance_epsilon=self.norm_eps)
        
        # Residual + Norm + MLP
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        # Map string to torch dtype (TPU v5e natively uses bfloat16)
        self.forward_dtype = torch.bfloat16 if config.forward_dtype == "bfloat16" else torch.float32

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head       = CastedLinear(config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = (-(config.puzzle_emb_ndim // -config.hidden_size) 
                               if config.puzzle_emb_len == 0 else config.puzzle_emb_len)
        
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                                                    batch_size=config.batch_size, init_std=0, cast_to=self.forward_dtype)

        if config.pos_encodings == "rope":
            # Optimization: pass xm.xla_device() if available, else standard init
            self.rotary_emb = RotaryEmbedding(dim=config.hidden_size // config.num_heads,
                                              max_position_embeddings=config.seq_len + self.puzzle_emb_len,
                                              base=config.rope_theta)
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(config.seq_len + self.puzzle_emb_len, config.hidden_size, 
                                             init_std=embed_init_std, cast_to=self.forward_dtype)

        # Build Reasoning Layers
        self.layers = nn.ModuleList([TinyRecursiveReasoningModel_ACTV1Block(self.config) for _ in range(self.config.L_layers)])

        # TPU Initialization: Using nn.Buffer ensures these are part of the model state and move to TPU
        self.register_buffer("H_init", trunc_normal_init_(torch.empty(config.hidden_size), std=1).to(self.forward_dtype))
        self.register_buffer("L_init", trunc_normal_init_(torch.empty(config.hidden_size), std=1).to(self.forward_dtype))

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            
            # Reshape is safer than view for XLA
            puzzle_embedding = puzzle_embedding.reshape(-1, self.puzzle_emb_len, self.config.hidden_size)
            embedding = torch.cat((puzzle_embedding, embedding), dim=-2)

        if self.config.pos_encodings == "learned":
            # 1/sqrt(2) approx = 0.7071
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, device):
        # TPU Optimization: create tensors directly on the active device
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.zeros(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L=torch.zeros(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        # torch.where is extremely efficient in XLA graphs
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.reshape(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.reshape(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple:
        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        z_H, z_L = carry.z_H, carry.z_L
        
        # Adaptive loops: XLA will unroll these into a static compute graph
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    for layer in self.layers:
                        z_L = layer(cos_sin, z_L + z_H + input_embeddings)
                for layer in self.layers:
                    z_H = layer(cos_sin, z_H + z_L)

        # Final step with gradient tracking
        for _ in range(self.config.L_cycles):
            for layer in self.layers:
                z_L = layer(cos_sin, z_L + z_H + input_embeddings)
        for layer in self.layers:
            z_H = layer(cos_sin, z_H + z_L)

        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) 
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])

class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size, device),
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),
            current_data={k: torch.zeros_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple:
        # 1. Reset carry for halted sequences
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        # 2. Update state tensors using where logic (very TPU friendly)
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)
        
        halt_mask = carry.halted.reshape((-1, ) + (1, ) * (batch["inputs"].ndim - 1))
        new_current_data = {}
        for k in batch.keys():
            new_current_data[k] = torch.where(halt_mask, batch[k], carry.current_data[k])

        # 3. Inner model forward
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        # 4. Halting Logic (ACT)
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            # Calculate halting flag based on Q-values
            if self.config.no_ACT_continue:
                halted = is_last_step | (q_halt_logits > 0)
            else:
                halted = is_last_step | (q_halt_logits > q_continue_logits)

            # Exploration Logic
            if self.training and self.config.halt_max_steps > 1:
                explore_mask = torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                min_halt_steps = explore_mask * torch.randint(2, self.config.halt_max_steps + 1, q_halt_logits.shape, device=q_halt_logits.device)
                halted = halted & (new_steps >= min_halt_steps)

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs