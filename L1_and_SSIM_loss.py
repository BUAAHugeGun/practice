import torch
import math
import sys
import torch.nn as nn
import Lp_Loss
import SSIM_Loss


class loss(nn.Module):
    def __init__(self, average=True, SSIM_c1=0.2, SSIM_c2=0.2, SSIM_c3=0.1, arfa=1, beta=1, gamma=1, k_L1=1, k_SSIM=1):
        super(loss, self).__init__()
        self.lp_Loss=Lp_Loss.Loss(average=average)
        self.ssim_Loss=SSIM_Loss.Loss(SSIM_c1,SSIM_c2,SSIM_c3,arfa,beta,gamma)
        self.k_L1=k_L1
        self.k_SSIM=k_SSIM

    def forward(self, input, target):
        return self.lp_Loss(input, target) * self.k_L1 + self.ssim_Loss(input, target) * self.k_SSIM


if __name__ == "__main__":
    x = torch.randn([10, 10])
    y = torch.randn([10, 10])
    z = torch.randn([9, 10])
    test = loss()
    print(test(x, x))
    print(test(x, y))
