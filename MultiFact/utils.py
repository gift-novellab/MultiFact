from typing import Optional

import torch
from torch import Tensor


EPS = torch.finfo(torch.float32).eps


def normalize_nonnegative(
    x: Tensor,
    dim: int,
    eps: float = 1e-12,
) -> Tensor:
    return x / x.sum(dim=dim, keepdim=True).clamp_min(eps)


def prepare_target(X: Tensor, device: Optional[torch.device] = None) -> Tensor:
    """Convert cells x genes expression into gene distributions per cell."""

    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    if X.ndim != 2:
        raise ValueError("X must have shape (cells, genes).")
    if torch.any(X < 0):
        raise ValueError("X must be non-negative.")
    return normalize_nonnegative(X.t().contiguous(), dim=0)
