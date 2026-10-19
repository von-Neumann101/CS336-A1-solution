from einops import einsum
import torch
from torch import nn

class RMSNorm(torch.nn.Module):
    def __init__(self, 
                 d_model: int, 
                 eps: float = 1e-5, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_dtype = x.dtype
        x = x.to(torch.float32)

        mean_square_x = einsum(
            x,
            x,
            "... d, ... d -> ..."
        ).unsqueeze(-1) / self.d_model #使用 unsqueeze 扩展维度
        RMS = torch.sqrt(mean_square_x + self.eps)
        RMSNorm = einsum(
            x,
            self.g,
            "... d, d -> ... d"
        ) / RMS

        return RMSNorm.to(origin_dtype)
