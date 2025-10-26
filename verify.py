import argparse

import torch

import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Verify GEMM implementation")
    parser.add_argument(
        "--M", type=int, default=128, help="Number of rows in matrix A and C"
    )
    parser.add_argument(
        "--N", type=int, default=128, help="Number of columns in matrix B and C"
    )
    parser.add_argument(
        "--K", type=int, default=128, help="Number of columns in matrix A and rows in matrix B"
    )
    parser.add_argument(
        "--dtype", type=str, default="float32", help="Data type for the tensors"
    )
    parser.add_argument(
        "--threshold", type=float, default=1e-5, help="Threshold for tensor consistency check"
    )
    parser.add_argument(
        "--kernel", type=str, default="gemm", help="Kernel to verify"
    )
    parser.add_argument(
        "--list_kernels", action="store_true", help="List all registered kernels"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.list_kernels:
        print("Registered kernels:")
        for name in utils.get_registered_kernels():
            print(f"- {name}")
        return

    M, N, K = args.M, args.N, args.K
    dtype = utils.get_torch_dtype(args.dtype)
    A = torch.randn(M, K).to(dtype=dtype, device='cuda')
    B = torch.randn(K, N).to(dtype=dtype, device='cuda')
    C = torch.zeros(M, N).to(dtype=dtype, device='cuda')

    kernel_func = utils.get_kernel(args.kernel)
    if kernel_func is None:
        print(f"Kernel '{args.kernel}' not found. Use --list_kernels to see available kernels.")
        return

    kernel_func(A, B, C)

    C_ref = torch.matmul(A, B)

    try:
        consistent, difference = utils.check_tensor_consistency(C_ref, C, threshold=args.threshold)
        if consistent:
            print(f"[Pass] {args.kernel} - Max difference({difference.item()}) <= Threshold ({args.threshold})!")
        else:
            print(f"[Fail] {args.kernel} - Max difference({difference.item()}) > Threshold ({args.threshold})!")
    except ValueError as e:
        print(f"Verification error: {e}")


if __name__ == "__main__":
    main()
