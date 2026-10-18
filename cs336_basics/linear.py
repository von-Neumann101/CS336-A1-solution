import math

from einops import einsum
import torch
from torch import Tensor, nn
class Linear(torch.nn.Module):
    def __init__(self, in_features: int, 
                 out_features: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None
                 ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(
                        torch.empty(self.out_features, self.in_features, device=device, dtype=dtype)
                        )
        sigma = math.sqrt(2 / (in_features + out_features))

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=sigma,
            a=-3 * sigma,
            b=3 * sigma
)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = einsum(
            x, 
            self.weight,
            "... in_features, out_features in_features -> ... out_features"
        )

        return output
    