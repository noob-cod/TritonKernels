def get_qmin_qmax(num_bits: int, symmetric: bool) -> tuple[int, int]:
    if symmetric:
        qmin = -(2 ** (num_bits - 1)) + 1
        qmax = (2 ** (num_bits - 1)) - 1
    else:
        qmin = 0
        qmax = (2 ** num_bits) - 1
    return qmin, qmax
