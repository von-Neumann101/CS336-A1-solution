import torch


def cross_entropy(o: torch.Tensor, x: torch.Tensor):
    """
    Args:
        o: 预测的logits
        x: 目标
    """
    o_max = torch.max(o, dim=-1, keepdim=True).values
    o_sum = torch.sum(torch.exp(o - o_max), dim=-1, keepdim=True)

    log_softmax = (torch.gather(o, dim=-1, index=x.unsqueeze(-1)) - o_max) - torch.log(o_sum)
    return -torch.mean(log_softmax)