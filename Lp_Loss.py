import torch
import math
import sys
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, average=True, p=1):
        super(Loss, self).__init__()
        self.average = average
        self.p = p

    def forward(self, input, target):
        if input.shape != target.shape:
            assert 0
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
            target = target.unsqueeze(0)
        ret = (torch.abs(input - target) ** self.p).sum() ** (1. / self.p)  # type: float
        if self.average == True:
            return ret / (input.shape[1] * input.shape[2])
        else:
            return ret


if __name__ == "__main__":
    x = torch.randn([10, 10])
    y = torch.randn([10, 10])
    z = torch.randn([9, 10])
    test = Loss()
    print(test(x, x))
    print(test(x, y))
