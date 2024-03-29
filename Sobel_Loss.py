import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
from Lp_Loss import Loss as lp_loss


class Loss(nn.Module):
    def __init__(self, arfa=0.5, beta=0.5, tp='float'):
        super(Loss, self).__init__()
        self.arfa = arfa
        self.beta = beta
        self.tp = tp

    def sobel(self, input):
        if isinstance(input, torch.Tensor):
            input = input.numpy()
        sobel_x = cv2.Sobel(input, cv2.CV_16S, 1, 0)
        sobel_y = cv2.Sobel(input, cv2.CV_16S, 0, 1)
        abs_x = cv2.convertScaleAbs(sobel_x)
        abs_y = cv2.convertScaleAbs(sobel_y)
        output = cv2.addWeighted(abs_x, self.arfa, abs_y, self.beta, 0)
        return output

    # cv2.imshow('output', output)
    # cv2.waitKey(0)

    def forward(self, input, target):
        if self.tp == 'float':
            input = (input * 256 + 0.5) // 1
            target = (target * 256 + 0.5) // 1
        input = self.sobel(input)
        target = self.sobel(target)
        Lp = lp_loss()
        return Lp(torch.from_numpy(input / 256), torch.from_numpy(target / 256))


if __name__ == '__main__':
    input = cv2.imread('./test.jpeg')
    # input = np.sum(input, 2)
    # input = (input + 0.5) // 3 / 256
    test = Loss()
    x = input.copy()
    for i in range(0, 400):
        temp = x[i].copy()
        x[i] = x[i + 400]
        x[i + 400] = temp
    # input = np.swapaxes(input, 0, 2)
    input = input / 256
    # x = np.swapaxes(x, 0, 2)
    x = x / 256
    print(test(input, x))
