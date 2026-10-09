"""
Fourier Neural Operator package for PDE solving.
"""

from .model import FNO1d, SpectralConv1d
from .dataset import BurgersDataset, generate_burgers_data

__all__ = [
    "FNO1d",
    "SpectralConv1d",
    "BurgersDataset",
    "generate_burgers_data",
]
