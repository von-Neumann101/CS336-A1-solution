from einops import rearrange
import torch
from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 max_seq_len: int | None = None,
                 theta: float = 10000.0,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.theta = theta
        self.max_seq_len = max_seq_len
        
        # d_k/v 指的是每一个头的 KV长度
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.rope = None
        if max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(self.theta, 
                                                self.d_k, 
                                                self.d_model,
                                                self.device)

        self.W_qkv = Linear(d_model, num_heads * (2 * self.d_k + self.d_v), device, dtype)
        
        self.W_o = Linear(num_heads * self.d_v, d_model, device, dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: 形状为(..., d_model)
        """
        Q, K, V = torch.split(self.W_qkv.forward(x), 
                              [self.d_k * self.num_heads, self.d_k * self.num_heads, self.d_v * self.num_heads], 
                              dim=-1)

        # 注意力头
        Qn = rearrange(
            Q,
            "... seq_len (h d_k) -> ... h seq_len d_k",
            h = self.num_heads
        )
        
        Kn = rearrange(
            K,
            "... seq_len (h d_k) -> ... h seq_len d_k",
            h = self.num_heads
        )
        
        Vn = rearrange(
            V,
            "... seq_len (h d_v) -> ... h seq_len d_v",
            h = self.num_heads
        )

        # Q, K 添加位置编码(rope)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    x.shape[-2],
                    device=x.device
                ).expand(*Qn.shape[:-1])

            Kn = self.rope.forward(Kn, token_positions)
            Qn = self.rope.forward(Qn, token_positions)

        mask = torch.tril(
            torch.ones(Q.shape[-2], K.shape[-2], dtype=torch.bool, device = Q.device),
        )

        raw = scaled_dot_product_attention(Qn, Kn, Vn, mask)

        raw = rearrange(
            raw,
            "... h seq_len d_v -> ... seq_len (h d_v)"
        )

        return self.W_o.forward(raw)

