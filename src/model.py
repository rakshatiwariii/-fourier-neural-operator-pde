import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """
    1D Fourier layer: Performs Fast Fourier Transform (FFT), linear transform 
    on lower modes in frequency domain, and Inverse FFT back to physical domain.
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to retain

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def _compl_mul1d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (batch, in_channel, x) x (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]

        # 1. Real Fast Fourier Transform along spatial domain
        x_ft = torch.fft.rfft(x)

        # 2. Multiply lower modes with complex weight parameters
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat
        )
        out_ft[:, :, :self.modes1] = self._compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # 3. Inverse Real Fast Fourier Transform
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator mapping initial PDE condition u0(x) -> u(x, T).
    """
    def __init__(self, modes: int = 16, width: int = 64):
        super(FNO1d, self).__init__()
        self.modes = modes
        self.width = width

        # Input projection layer: [u0(x), x_coordinate] (2 channels) -> hidden width
        self.fc0 = nn.Linear(2, self.width)

        # Fourier Spectral Convolution Layers
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)

        # Skip-connection spatial convolutions
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)

        # Output projection layers: hidden width -> 128 -> 1 channel
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Append 1D grid representation
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        # Lift features
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        # Fourier blocks with GELU activations
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))

        # Project back to physical domain
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_grid(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(torch.linspace(0, 1, size_x), dtype=torch.float32)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
