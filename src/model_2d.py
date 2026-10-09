import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution Layer.
    Performs 2D Fast Fourier Transform (FFT), applies complex matrix multiplication
    on lower spatial frequency modes, and transforms back via Inverse 2D FFT.
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Fourier modes retained along height (X)
        self.modes2 = modes2  # Fourier modes retained along width (Y)

        self.scale = 1.0 / (in_channels * out_channels)
        
        # Complex weights for top-left and bottom-left spatial frequency quadrants
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def _compl_mul2d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # Complex tensor contraction across 2D spatial modes
        # (batch, in_channel, x, y) x (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]

        # 1. Perform 2D Real Fast Fourier Transform
        x_ft = torch.fft.rfft2(x)

        # 2. Allocate output frequency tensor
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # 3. Filter and multiply low-frequency corners
        out_ft[:, :, :self.modes1, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # 4. Inverse 2D Real Fast Fourier Transform
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator mapping initial fluid vorticity w0(x, y) -> w(x, y, T)
    for 2D incompressible Navier-Stokes flow.
    """
    def __init__(self, modes1: int = 12, modes2: int = 12, width: int = 32):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # Input feature projection: [w0(x, y), x_coord, y_coord] (3 channels) -> width
        self.fc0 = nn.Linear(3, self.width)

        # 4 Spectral Convolution Blocks
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        # 1x1 2D Convolutions for residual connections
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # Output projections back to scalar physical domain output
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Append 2D spatial grid coordinate mesh
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        # Lift features
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # Spectral blocks + Skip connections
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = F.gelu(self.conv3(x) + self.w3(x))

        # Project features
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_grid(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(torch.linspace(0, 1, size_x), dtype=torch.float32)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        
        gridy = torch.tensor(torch.linspace(0, 1, size_y), dtype=torch.float32)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        
        return torch.cat((gridx, gridy), dim=-1).to(device)
