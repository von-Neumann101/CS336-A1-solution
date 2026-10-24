import numpy as np
import torch

def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 随机采样 batch_size 个起始位置
    starts = np.random.randint(
        0, len(x) - context_length, size=batch_size
    )

    # 构造索引矩阵，形状 (batch_size, context_length)
    indices = starts[:, None] + np.arange(context_length)
    inputs = torch.as_tensor(x[indices], dtype=torch.long, device=device)
    targets = torch.as_tensor(x[indices + 1], dtype=torch.long, device=device)

    return inputs, targets