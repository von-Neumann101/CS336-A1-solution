from collections.abc import Callable, Iterable
import math
from typing import Any, Optional

import torch

"""
param_groups: 有哪些参数需要更新？每组参数使用什么超参数？
state: 每个参数在优化过程中积累了哪些历史信息？
"""

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if not (lr >= 0 and 0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError("超参数不合法")
        
        defaults = {"lr": lr, 
                    "betas": betas, 
                    "eps": eps, 
                    "weight_decay": weight_decay} # 各参数组使用的默认超参数字典
        
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups: # param_groups 是字典列表，保存参数的优化器参数
                    lr, betas, eps = group["lr"], group["betas"], group["eps"]
                    weight_decay = group["weight_decay"]

                    for p in group["params"]:
                        if p.grad is None:
                            continue

                        state = self.state[p]  # 获取与 p 关联的状态
                        if not state:
                            state["m"] = torch.zeros_like(p)
                            state["v"] = torch.zeros_like(p)
                            state["t"] = 1
                        
                        m, v, t = state["m"], state["v"], state["t"]   
                        grad = p.grad.data  # 获取损失函数对 p 的梯度
                        lr_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                        p.data *= 1 - weight_decay * lr
                        m = betas[0] * m + (1 - betas[0]) * grad
                        v = betas[1] * v + (1 - betas[1]) * (grad ** 2)
                        p.data -= lr_t * m / (torch.sqrt(v) + eps)
                        state["m"] = m
                        state["v"] = v
                        state["t"] = t + 1  # 递增迭代次数

        return loss