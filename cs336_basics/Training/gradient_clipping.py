from collections.abc import Iterable

import torch


def gradient_clipping(
        parameters: Iterable[torch.nn.Parameter], 
        max_l2_norm: float,
        eps = 1e-6
):
    l2 = 0
    grad_list = []
    for p in parameters:
        if p.grad is None:
            continue

        grad_list.append(p.grad)
        l2 += torch.sum(p.grad ** 2)

    if l2 == 0:
        return None
    
    l2_norm = torch.sqrt(l2)

    if l2_norm >= max_l2_norm:
        torch._foreach_mul_(grad_list, max_l2_norm / (l2_norm + eps))
    
    return grad_list
        