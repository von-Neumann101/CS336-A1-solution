import torch

from cs336_basics.Model.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.Model.rms_norm import RMSNorm
from cs336_basics.Model.swiglu import SwiGLU

class TransformerBlock(torch.nn.Module):
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 eps: float = 1e-5, 
                 theta: float = 10000.0,
                 max_seq_len: int | None = None,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        """
        Args:
            d_model: Transformer Block 输入的维度。
            num_heads: 多头自注意力使用的注意力头数量。
            d_ff: 逐位置前馈网络内部隐藏层的维度
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.mha = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, device, dtype)
        self.mha_RMS = RMSNorm(d_model, eps, device, dtype)
        self.ffn_RMS = RMSNorm(d_model, eps, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x):
        h1 = x + self.mha.forward(self.mha_RMS.forward(x))
        return h1 + self.ffn.forward(self.ffn_RMS.forward(h1))