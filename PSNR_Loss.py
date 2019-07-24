import torch
import torch.nn as nn
import numpy as np
import torchvision
import cv2

import math


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, input, target):
        if input.shape != target.shape:
            assert 0
        MSE = ((target - input) ** 2).sum() / (input.shape[0] * input.shape[1])
        return 10 * math.log10(1. / MSE)


if __name__ == '__main__':
    input = cv2.imread('./test.jpeg')
    input = np.sum(input, 2)
    input = (input + 0.5) // 3 / 256
    test = Loss()
    x = input.copy()
    for i in range(0, 1):
        temp = x[i].copy()
        x[i] = x[i + 400]
        x[i + 400] = temp
    print(test(input, x))
