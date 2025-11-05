from typing import Any, Callable, List, Optional

import torch

from tk import logger
import tk.utils as utils


GEMM_FUNC_SIGNATURE = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

class GemmVerifier:

    def __init__(
        self, 
        golden_gemm: GEMM_FUNC_SIGNATURE,
        target_gemm: GEMM_FUNC_SIGNATURE | List[GEMM_FUNC_SIGNATURE],
        golden_device: str = 'cuda',
        target_device: str = 'cuda',
    ) -> None:
        self.golden_gemm = golden_gemm
        if not isinstance(target_gemm, list):
            target_gemm = [target_gemm]
        self.target_gemm_list = target_gemm
        self.golden_device = golden_device
        self.target_device = target_device

    @classmethod
    def _invoke_with_dummy_inputs(
        cls,
        gemm_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        A: torch.Tensor, B: torch.Tensor,
        device: str = 'cuda',
    ) -> torch.Tensor:
        C = gemm_op(A.to(device), B.to(device))
        return C

    def _verify_single(
        self,
        golden_op: GEMM_FUNC_SIGNATURE,
        target_op: GEMM_FUNC_SIGNATURE,
        verify_fn: Callable[[torch.Tensor, torch.Tensor], Any],
        M: int, N: int, K: int,
        dtype: torch.dtype = torch.float32,
    ) -> str:
        if dtype.is_floating_point:
            A = torch.randn((M, K), dtype=dtype)
            B = torch.randn((K, N), dtype=dtype)
        else:
            A = torch.randint(-128, 127, (M, K), dtype=dtype)
            B = torch.randint(-128, 127, (K, N), dtype=dtype)

        golden = self._invoke_with_dummy_inputs(golden_op, A, B, device=self.golden_device)
        target = self._invoke_with_dummy_inputs(target_op, A, B, device=self.target_device)

        if golden.device != target.device:
            target = target.to(golden.device)

        return verify_fn(target, golden)

    def verify(
        self,
        test_cases: list[tuple[int, int, int]],
        dtype: torch.dtype = torch.float32,
        verify_fn: Optional[Callable[[torch.Tensor, torch.Tensor], str]] = None,
    ) -> None:
        """ Verify all target GEMM implementations against the golden implementation.
        Args:
            test_cases (list[tuple[int, int, int]]): List of (M, N, K) tuples for GEMM sizes to test.
            dtype (torch.dtype): Data type for the GEMM operations.
            verify_fn (Optional[Callable[[torch.Tensor, torch.Tensor], str]]): 
                Optional function to verify two tensors, return the conclusion string of verification. 
                If None, a basic verify function is used.
        """
        if verify_fn is None:
            dtype_str = utils.convert_torch_dtype_to_string(dtype)
            def basic_verify_fn(a: torch.Tensor, b: torch.Tensor) -> str:
                is_equal = torch.allclose(a, b, atol=1e-5, rtol=1e-3)
                res = "PASS" if is_equal else "FAIL"
                conclusion = f" - [{res}] {dtype_str:^7s}"
                return conclusion
            verify_fn = basic_verify_fn
        for target_op in self.target_gemm_list:
            logger.info("-" * 50)
            logger.info(f"Verifying target GEMM: {target_op.__name__}")
            for M, N, K in test_cases:
                conclusion: str = self._verify_single(
                    self.golden_gemm, target_op,
                    verify_fn,
                    M, N, K,
                    dtype=dtype,
                )
                logger.info(f"{conclusion}, GEMM Size M={M}, N={N}, K={K}")
