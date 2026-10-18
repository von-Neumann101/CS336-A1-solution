import math

import torch
from torch import nn

class Embedding(torch.nn.Module):
    def __init__(
            self, 
            num_embeddings: int, 
            embedding_dim: int, 
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
            ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(
                                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
                                )
        
        sigma = math.sqrt(2 / (num_embeddings + embedding_dim))
        nn.init.trunc_normal_(
                    self.weight,
                    mean=0.0,
                    std=sigma,
                    a=-3 * sigma,
                    b=3 * sigma
                )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


