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

        position = torch.arange(max_seq_len, device=device)
        j = torch.arange(0, d_k // 2, device=device)
        frequency = theta ** (-2 * j / d_k)

        angle = einsum(
                    position,
                    frequency,
                    "pos, pair -> pos pair"
                )

        sin = torch.sin(angle) # i, j
        cos = torch.cos(angle)

        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = rearrange(
            x,
            "... (pair two) -> ... pair two",
            two=2
        )

        # 先广播，这样可以使得内存占用少 h 倍
        token_positions = rearrange(
            token_positions,
            "... seq -> ... h seq",
            h=1
        )

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        raw1 = einsum(
            x[..., 0],
            cos,
            "... seq pos, ... seq pos -> ... seq pos"
        ) + einsum(
            x[..., 1],
            -sin,
            "... seq pos, ... seq pos -> ... seq pos"
        )

        raw2 = einsum(
            x[..., 0],
            sin,
            "... seq pos, ... seq pos -> ... seq pos"
        ) + einsum(
            x[..., 1],
            cos,
            "... seq pos, ... seq pos -> ... seq pos"
        )

        raw = rearrange(
            [raw1, raw2],
            "two ... -> ... two"
        )

        return rearrange(
            raw,
            "... pair two -> ... (pair two)"
        )
