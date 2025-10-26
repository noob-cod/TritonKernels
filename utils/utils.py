import torch


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """ Convert a string representation of a data type to a torch.dtype.
    Args:
        dtype_str (str): The string representation of the data type.
    Returns:
        torch.dtype: The corresponding torch.dtype.
    Raises:
        ValueError: If the provided string does not correspond to a valid torch dtype.
    """
    alias_map = {
        "half": "float16",
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "u8": "uint8",
        "b": "bool",
    }
    if dtype_str in alias_map:
        dtype_str = alias_map[dtype_str]
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")
    return dtype_map[dtype_str]
