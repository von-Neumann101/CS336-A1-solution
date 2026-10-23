from sympy import Float
import torch


def softmax(x, d):
    C = torch.max(input=x, dim=d, keepdim=True).values
    exp_x = torch.exp(x - C)
    return exp_x / torch.sum(input=exp_x, dim=d, keepdim=True)
    
