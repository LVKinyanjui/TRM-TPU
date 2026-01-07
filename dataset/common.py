import math
from typing import List, Optional
import torch
from torch import nn
import numpy as np
import pydantic

# --- DATASET METADATA ---

class PuzzleDatasetMetadata(pydantic.BaseModel):
    pad_id: int
    ignore_label_id: Optional[int]
    blank_identifier_id: int
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    total_puzzles: int
    sets: List[str]

# --- GEOMETRIC TRANSFORMATIONS (CPU/NumPy) ---

# Global list mapping each dihedral transform id to its inverse.
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]

def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries by rotate, flip and mirror (NumPy/CPU)"""
    if tid == 0: return arr
    elif tid == 1: return np.rot90(arr, k=1)
    elif tid == 2: return np.rot90(arr, k=2)
    elif tid == 3: return np.rot90(arr, k=3)
    elif tid == 4: return np.fliplr(arr)
    elif tid == 5: return np.flipud(arr)
    elif tid == 6: return arr.T
    elif tid == 7: return np.fliplr(np.rot90(arr, k=1))
    else: return arr

def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])

# --- WEIGHT INITIALIZATION (XLA/TPU Compatible) ---

def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    """
    Precision-correct truncated normal initialization (JAX-style).
    Ensures that the actual variance of the initialized weights matches 'std'.
    """
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            # Constants calculated on CPU host
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            
            # Variance correction factor
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            # XLA-compatible sequence of operations
            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            # clamp_ is highly optimized in XLA
            tensor.clamp_(lower * comp_std, upper * comp_std)

    return tensor