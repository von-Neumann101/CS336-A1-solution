from einops import einsum
import torch

from cs336_basics.linear import Linear

class SwiGLU(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.linear1 = Linear(d_model, d_ff, device, dtype)
        self.linear2 = Linear(d_ff, d_model, device, dtype)
        self.linear3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l1 = self.linear1.forward(x)
        h1 = l1 * torch.sigmoid(l1)

        h3 = self.linear3.forward(x)

        h2 = einsum(
            h1,
            h3,
            "... dff, ... dff -> ... dff"
        )

        return self.linear2.forward(h2)