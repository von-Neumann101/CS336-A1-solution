import torch
from torch import nn

from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rms_norm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock

class TransformerLM(torch.nn.Module):
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 eps: float = 1e-5, 
                 theta: float = 10000.0,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                eps=eps,
                theta=theta,
                max_seq_len=context_length,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        self.emb = Embedding(vocab_size, d_model, device, dtype)
        self.rms = RMSNorm(d_model, eps, device, dtype)
        self.linear = Linear(d_model, vocab_size, device, dtype)

    def forward(self, x):
        x = self.emb.forward(x)
        for block in self.layers:
            x = block.forward(x)
        return self.linear(self.rms(x))