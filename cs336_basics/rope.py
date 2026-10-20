from einops import einsum, rearrange
import torch


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, 
                 theta: float, 
                 d_k: int, 
                 max_seq_len: int, 
                 device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        position = torch.arange(max_seq_len)
        j = torch.arange(0, d_k // 2, device=device)
        frequency = theta ** (-2 * j / d_k)


        self.register_buffer("angle", 
                             einsum(
                                position,
                                frequency,
                                "pos, pair -> pos pair"
                             ),
                             persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = rearrange(
            x,
            "... (pair two) -> ... pair two",
            two=2
        )

        angle = self.angle[token_positions]
        
        sin = torch.sin(angle) # i, j
        cos = torch.cos(angle)

        R1 = rearrange(
            [cos, -sin],
            "two ... seq pos -> ... seq pos two"
        )

        R2 = rearrange(
            [sin, cos],
            "two ... seq pos -> ... seq pos two"
        )

        raw1 = einsum(
            x,
            R1,
            "... seq pos two, ... seq pos two -> ... seq pos"
        ) 
        
        raw2 = einsum(
            x,
            R2,
            "... seq pos two, ... seq pos two -> ... seq pos"
        )

        raw = rearrange(
            [raw1, raw2],
            "two ... -> ... two"
        )

        return rearrange(
            raw,
            "... pair two -> ... (pair two)"
        )
