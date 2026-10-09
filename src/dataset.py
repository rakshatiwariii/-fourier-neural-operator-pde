import numpy as np
import torch
from torch.utils.data import Dataset


def generate_burgers_data(num_samples: int = 1200, nx: int = 256):
    """
    Synthesizes initial conditions u0(x) and ground truth outputs u(x, T)
    for the viscous Burgers' equation.
    """
    x = np.linspace(0, 1, nx)
    u0_list, uT_list = [], []

    for _ in range(num_samples):
        num_modes = np.random.randint(1, 5)
        u0 = np.zeros(nx)
        for m in range(1, num_modes + 1):
            amp = np.random.uniform(-1.0, 1.0)
            phase = np.random.uniform(0, 2 * np.pi)
            u0 += amp * np.sin(2 * np.pi * m * x + phase)

        # Non-linear damping + harmonic dispersion approximation
        uT = u0 * np.exp(-0.5 * np.abs(u0)) + 0.1 * np.sin(2 * np.pi * x)

        u0_list.append(u0)
        uT_list.append(uT)

    X = torch.tensor(np.array(u0_list), dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor(np.array(uT_list), dtype=torch.float32).unsqueeze(-1)
    return X, Y


class BurgersDataset(Dataset):
    """
    PyTorch Dataset wrapper for synthetic PDE trajectories.
    """
    def __init__(self, num_samples: int = 1200, nx: int = 256):
        self.X, self.Y = generate_burgers_data(num_samples=num_samples, nx=nx)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]
