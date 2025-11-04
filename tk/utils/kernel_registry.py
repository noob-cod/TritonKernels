KERNELS = {}

def register(name: str):
    def decorator(func):
        KERNELS[name] = func
        return func
    return decorator

def get_registered_kernels():
    return list(KERNELS.keys())

def get_kernel(name: str):
    return KERNELS.get(name, None)

from .. import (
    gemm, 
    quantization,
)  # Register all kernels
