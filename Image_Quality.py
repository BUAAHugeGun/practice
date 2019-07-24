import torch
import numpy as np
import torch.nn as nn
from Lp_Loss import Loss as lp_loss
from SSIM_Loss import Loss as ssim_loss
from PSNR_Loss import Loss as psnr_loss
from Sobel_Loss import Loss as sobel_loss
import cv2


class Estimate(nn.Module):
    def __init__(self):
        super(Estimate, self).__init__()
        self.space_set = ['RGB', 'YUV']

    def change_space(self, input, target, space):
        if space not in self.space_set:
            assert 0
        if space == 'YUV':
            input = input[0]
            target = target[0]

    def L1(self, input, target, space='RGB'):
        self.change_space(input, target, space)
        lp = lp_loss()
        return lp(input, target)

    def SSIM(self, input, target, space='RGB'):
        self.change_space(input, target, space)
        ssim = ssim_loss()
        return ssim(input, target)

    def PSNR(self, input, target, space='RGB'):
        self.change_space(input, target, space)
        psnr = psnr_loss()
        return psnr(input, target)

    def Sobel(self, input, target, space='RGB'):
        self.change_space(input, target, space)
        sobel = sobel_loss()
        if space == 'RGB':
            input = torch.sum(input, 0) / 3
            target = torch.sum(target, 0) / 3
        return sobel(input, target)


def toYUV(x):
    y = x.contiguous().view(-1, 3).double()
    mat = torch.tensor([[0.257, -0.148, 0.439],
                        [0.564, -0.291, -0.368],
                        [0.098, 0.439, -0.071]]).double()
    bias = torch.tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).double()
    y = y.mm(mat) + bias
    y = y.view(3, x.shape[1], x.shape[2])
    return y


if __name__ == '__main__':
    input = cv2.imread('./test.jpeg')
    # input = np.sum(input, 2)
    input = (input) / 256
    test = Estimate()
    x = input.copy()
    for i in range(0, 1):
        temp = x[i].copy()
        x[i] = x[i + 400]
        x[i + 400] = temp
    input = np.swapaxes(input, 0, 2)
    x = np.swapaxes(x, 0, 2)
    input = torch.from_numpy(input)
    x = torch.from_numpy(x)
    print(test.L1(input, x))
    print(test.SSIM(input, x))
    print(test.PSNR(input, x))
    print(test.Sobel(input, x))
    input = toYUV(input)
    x = toYUV(x)
    print(test.L1(input, x))
    print(test.SSIM(input, x))
    print(test.PSNR(input, x))
    print(test.Sobel(input, x))
