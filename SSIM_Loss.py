import torch
import math
import sys
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, SSIM_c1=0.2, SSIM_c2=0.2, SSIM_c3=0.1, arfa=1, beta=1, gamma=1, k_L1=1, k_SSIM=1):
        super(Loss, self).__init__()
        self.SSIM_c1 = SSIM_c1
        self.SSIM_c2 = SSIM_c2
        self.SSIM_c3 = SSIM_c3
        self.arfa = arfa
        self.beta = beta
        self.gamma = gamma
        self.k_L1 = k_L1
        self.k_SSIM = k_SSIM

    def SSIM(self,input,target):
        n_input = input.shape[0] * input.shape[1]
        n_target = target.shape[0] * target.shape[1]
        if n_input != n_target:
            sys.stderr.write("expected input size :{} but get size :{}".format(target.shape, input.shape))
            assert 0
        mu_x = input.sum() / n_input
        mu_y = target.sum() / n_target
        sigma_x = math.sqrt(((mu_x - input) ** 2).sum() / n_input)
        sigma_y = math.sqrt(((mu_y - input) ** 2).sum() / n_target)
        sigma2_xy = (input * target).sum() / n_input - mu_x * mu_y
        SSIM_L = (2. * mu_x * mu_x + self.SSIM_c1) / (mu_x * mu_x + mu_y * mu_y + self.SSIM_c1)
        SSIM_C = (2. * sigma2_xy + self.SSIM_c2) / (sigma_x * sigma_x + sigma_y * sigma_y + self.SSIM_c2)
        SSIM_S = (sigma2_xy + self.SSIM_c3) / (sigma_x * sigma_y + self.SSIM_c3)
        return (SSIM_L ** self.arfa) * (SSIM_C ** self.beta) * (SSIM_S ** self.gamma)

    def forward(self, input, target):
        return 1-self.SSIM(input,target)


if __name__ == "__main__":
    x = torch.randn([10, 10])
    y = torch.randn([10, 10])
    z = torch.randn([9, 10])
    test = Loss()
    print(test(x, x))
    print(test(x, y))
    print(test(x, z))
    print(test(x, z))
