import math
import torch
from torch import nn

def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    """
    JAX-style truncated normal initialization.
    Mathematically precise version that ensures the resulting variance 
    actually matches the requested 'std'.
    """
    if std == 0:
        with torch.no_grad():
            return tensor.zero_()

    # TPU/XLA Note: We perform the constant calculations in Python (math module).
    # These execute on the CPU host during model setup, which is efficient.
    
    sqrt2 = math.sqrt(2)
    # Convert bounds to standardized normal space
    a = math.erf(lower / sqrt2)
    b = math.erf(upper / sqrt2)
    z = (b - a) / 2

    # Normal PDF constants
    c = (2 * math.pi) ** -0.5
    pdf_u = c * math.exp(-0.5 * (lower ** 2))
    pdf_l = c * math.exp(-0.5 * (upper ** 2))
    
    # Calculate the correction factor for the variance of a truncated distribution
    denominator = math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
    comp_std = std / denominator

    with torch.no_grad():
        # 1. Fill with uniform noise in the transformed probability space [a, b]
        tensor.uniform_(a, b)
        
        # 2. Transform to normal space using the Inverse Error Function
        # XLA supports erfinv_ natively.
        tensor.erfinv_()
        
        # 3. Scale and Clamp
        # We use clamp_ instead of clip_ (aliases, but clamp is standard)
        tensor.mul_(sqrt2 * comp_std)
        tensor.clamp_(lower * comp_std, upper * comp_std)

    return tensor