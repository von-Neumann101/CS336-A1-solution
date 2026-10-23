import math

from einops import einsum
import torch

from cs336_basics.Model.softmax import softmax


def scaled_dot_product_attention(
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor,
        mask: torch.Tensor | None = None
) -> torch.Tensor:
    d_k = Q.shape[-1]

    attention_score = einsum(
        Q,
        K,
        "... seq_len1 d_k, ... seq_len2 d_k -> ... seq_len1 seq_len2"
    ) / math.sqrt(d_k)

    if mask is not None:
        attention_score = attention_score.masked_fill(~mask, -torch.inf)

    score = softmax(attention_score, -1)

    return einsum(
        score,
        V,
        "... seq_len1 seq_len2, ... seq_len2 d_v -> ... seq_len1 d_v"
    )

    
