import torch
import triton
import triton.language as tl


@triton.jit
def t_round_to_nearest_even(x):
    """Round to nearest even"""
    rounded = tl.floor(x + 0.5)
    is_half = (x % 1.0) == 0.5
    even_correction = (rounded % 2.0) == 1.0
    return tl.where(is_half & even_correction, rounded - 1.0, rounded)