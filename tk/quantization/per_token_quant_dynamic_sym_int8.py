from typing import Optional
from unittest import case

import torch
import triton
import triton.language as tl

from tk import logger
from .quant_utils import get_qmin_qmax
from .quant_triton_utils import t_round_to_nearest_even
from tk.utils.kernel_registry import register


@register("per_token_quant_dynamic_sym_int8")
def per_token_quant_dynamic_sym_int8(
    tensor: torch.Tensor,
):
    """
    Per-token dynamic symmetric int8 quantization.

    Args:
        tensor (torch.Tensor): Input tensor of shape (M, K) to be quantized.
    Returns:
        qtensor (torch.Tensor): Quantized tensor of shape (M, K) with int8 values.
        scales (torch.Tensor): Scale factors of shape (M,).
    """
    M, K = tensor.shape
    qtensor = torch.empty((M, K), dtype=torch.int8, device=tensor.device)
    scales = torch.empty((M, 1), dtype=torch.float32, device=tensor.device)

    BLOCK_M = 1
    BLOCK_K = K
    grid_m = (M + BLOCK_M - 1) // BLOCK_M

    per_token_quant_dynamic_sym_int8_kernel[(grid_m,)](
        tensor,
        scales,
        qtensor,
        tensor.stride(0),
        tensor.stride(1),
        scales.stride(0),
        qtensor.stride(0),
        qtensor.stride(1),
        M, K,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )

    return qtensor, scales


@triton.jit
def per_token_quant_dynamic_sym_int8_kernel(
    a_ptr, scale_ptr, qa_ptr,
    a_stride_m, a_stride_n,
    scale_stride_m,
    qa_stride_m, qa_stride_n,
    M: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    act = tl.load(
        a_ptr + (pid * BLOCK_M + offs_m[:, None]) * a_stride_m + offs_k[None, :] * a_stride_n,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        other=0.0
    )
    scale = tl.max(act, axis=1) / 127.0 + 1e-8

    q_act = tl.clamp(t_round_to_nearest_even(act / scale[:, None]), -128.0, 127.0)

    tl.store(
        qa_ptr + (pid * BLOCK_M + offs_m[:, None]) * qa_stride_m + offs_k[None, :] * qa_stride_n,
        q_act,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K)
    )
    tl.store(
        scale_ptr + (pid * BLOCK_M + offs_m) * scale_stride_m,
        scale,
        mask=(offs_m < M)
    )
