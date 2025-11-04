import torch


def check_tensor_consistency(
    golden: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 1e-5,):
    """ Check if two tensors are consistent within a given threshold. 
    Args:
        golden (torch.Tensor): The reference tensor.
        target (torch.Tensor): The tensor to compare against the reference.
        threshold (float): The maximum allowed difference between the tensors.
    Returns:
        bool: True if the tensors are consistent, False otherwise.
        if golden.shape != target.shape:
            raise ValueError(f"Shape mismatch: golden shape {golden.shape} vs target shape {target.shape}"
        if golden.dtype != target.dtype:
            raise ValueError(f"Dtype mismatch: golden dtype {golden.dtype} vs target dtype {target.dtype}"
    """
    if golden.shape != target.shape:
        raise ValueError(f"Shape mismatch: golden shape {golden.shape} vs target shape {target.shape}")
    if golden.dtype != target.dtype:
        raise ValueError(f"Dtype mismatch: golden dtype {golden.dtype} vs target dtype {target.dtype}")
    difference = torch.abs(golden - target).max()
    return difference <= threshold, difference
