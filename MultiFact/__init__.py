"""Sequential cell, condition, and batch factorization."""

from .config import MultiFactConfig
from .model import MultiFactModel
from .train import fit_multifact

__all__ = ["MultiFactConfig", "MultiFactModel", "fit_multifact"]
__version__ = "0.1.0"
