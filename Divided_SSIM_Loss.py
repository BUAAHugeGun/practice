import sys
import torch
import torch.nn as nn
import cv2 as cv
import os
import numpy as np
from Lp_Loss import Loss as L1


class Loss(nn.Module):
    def __init__(self, SSIM_c1=0.2, k_Loss=[9, 8, 7, 6, 5, 4, 3, 2, 1]):
        super(Loss, self).__init__()
        self.SSIM_c1 = SSIM_c1
        self.k_Loss = k_Loss

    def SSIM(self, input, target):
        if input.shape != target.shape:
            sys.stderr.write("expected input shape:{}, but get shape:{}".format(target.shape, input.shape))
            assert 0
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
            target = target.unsqueeze(0)
        n = input.shape[0] * input.shape[1]
        mu_x = input.sum() / n
        mu_y = target.sum() / n
        SSIM_L = (2. * mu_x * mu_y + self.SSIM_c1) / (mu_x * mu_x + mu_y * mu_y + self.SSIM_c1)
        return SSIM_L

    def SSIM_Loss(self, input, target):
        return 1 - self.SSIM(input, target)

    def calc_dfs(self, input, target, deep):
        ret = self.SSIM_Loss(input, target) * self.k_Loss[deep]
        if input.shape[0] == 1:
            return ret
        mid = (input.shape[0] + 1) // 2
        l = input.shape[0]
        ret += 0.25 * (
                self.calc_dfs(input[0:mid, 0:mid], target[0:mid, 0:mid], deep + 1) +
                self.calc_dfs(input[0:mid, mid:l], target[0:mid, mid:l], deep + 1) +
                self.calc_dfs(input[mid:l, 0:mid], target[mid:l, 0:mid], deep + 1) +
                self.calc_dfs(input[mid:l, mid:l], target[mid:l, mid:l], deep + 1)
        )
        return ret

    def forward(self, input, target):
        if input.shape[0] != input.shape[1]:
            sys.stderr.write("expected input shape[0] == shape[1], but get shape:{}".format(input.shape))
            assert 0
        return self.calc_dfs(input, target, 0)


if __name__ == "__main__":
    test = Loss()
    img = cv.imread(os.path.abspath('./test.jpeg'), 1)
    img = np.swapaxes(img, 0, 2)
    x = (img[:][0] + img[:][1] + img[:][2]) / 3 / 256
    input = x[5:261, 5:261]
    target = x[15:271, 15:271]
    input = torch.from_numpy(input)
    target = torch.from_numpy(target)
    print(test(input, target))
    l1 = L1()
    print(l1(input, target))

"""
(1,1)     0.003774832937427805 
          0.0068
(2,2)     0.009734030721230528
(3,3)     0.016439832254887495
(1,0)     0.0021855419626039463
          0.0049
(2,0)     0.006031101434479597
(3,0)     0.010280216363325222
(10,10)   0.07510099872167494
          0.0274
(10,0)    0.047085278608844
          0.0219
(100,100) 1.0747443688456522
(100,0)   0.30389802110195063
another erea 1.2334625421996155
             0.0867
"""
