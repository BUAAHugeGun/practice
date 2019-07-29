import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class DividedSsimLoss(nn.Module):
    def __init__(self, SSIM_c1=0.2, k_Loss=[9, 8, 7, 6, 5, 4, 3, 2, 1]):
        super(DividedSsimLoss, self).__init__()
        self.SSIM_c1 = SSIM_c1
        self.k_Loss = k_Loss
        self.rgb2gray = torch.FloatTensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        self.min_size = 256

    def SSIM(self, input, target):
        mu_x = input.mean(dim=[2, 3])
        mu_y = target.mean(dim=[2, 3])
        SSIM_L = (2. * mu_x * mu_y + self.SSIM_c1) / (mu_x ** 2 + mu_y ** 2 + self.SSIM_c1)
        return SSIM_L.mean()

    def SSIM_Loss(self, input, target):
        return 1 - self.SSIM(input, target)

    def calc_dfs(self, input, target):
        ret = 0.
        deep = 0
        while True:
            if len(input.shape) == 3:
                input = input.unsqueeze(1)
                target = target.unsqueeze(1)
            B, H, W = input.shape[1:]
            ret += self.SSIM_Loss(input, target) * self.k_Loss[deep]
            if H == 1 or W == 1:
                return ret
            mid = (H + 1) // 2
            input = torch.cat(
                [input[:, :, :mid, :mid], input[:, :, :mid, mid:],
                 input[:, :, mid:, :mid], input[:, :, mid:, mid:]])
            target = torch.cat(
                [target[:, :, :mid, :mid], target[:, :, :mid, mid:], target[:, :, mid:, :mid],
                 target[:, :, mid:, mid:]])
            deep += 1

    def calc_dfs_from_bottom_to_up(self, input, target):
        def ssim_layer(x, y):
            xx = x**2
            yy = y**2
            xy = x*y
            ans = (2*xy + self.SSIM_c1) / (xx + yy + self.SSIM_c1)
            return 1 - ans.mean()

        input = input.unsqueeze(1)
        target = target.unsqueeze(1)
        ret = 0.
        for i in range(9):
            ret += ssim_layer(input, target) * self.k_Loss[-(i+1)]
            input = F.avg_pool2d(input, kernel_size=2, stride=2)
            target = F.avg_pool2d(target, kernel_size=2, stride=2)

        return ret

    def forward(self, input, target, type='from_bottom_to_up'):
        assert (input.shape == target.shape)
        cal_func_dict = {'from_top_to_down': self.calc_dfs, 'from_bottom_to_up': self.calc_dfs_from_bottom_to_up}
        assert (type in cal_func_dict.keys())
        cal_func = cal_func_dict[type]

        B, C, H, W = input.shape
        L = self.min_size
        assert (H >= L and W >= L)

        pad_h = -H % L
        pad_w = -W % L
        input_pad = F.pad(input, [0, pad_h, 0, pad_w], mode='reflect')
        target_pad = F.pad(target, [0, pad_h, 0, pad_w], mode='reflect')

        if C == 3:
            device = torch.get_device(input) if 'cuda' in input.type() else 'cpu'
            self.rgb2gray = self.rgb2gray.to(device)
            input_gray = (input_pad * self.rgb2gray).sum(dim=1).squeeze(1)
            target_gray = (target_pad * self.rgb2gray).sum(dim=1).squeeze(1)
        else:
            input_gray = input_pad.squeeze(1)
            target_gray = target_pad.squeeze(1)

        B, H, W = input_gray.shape
        loss_tile = []
        for h in range(0, H, L):
            for w in range(0, W, L):
                old = time.time()
                loss_tile.append(cal_func(input_gray[:, h:h+L, w:w+L], target_gray[:, h:h+L, w:w+L]))
                print(time.time() - old)
        return sum(loss_tile) / len(loss_tile)


if __name__ == "__main__":
    criterion = DividedSsimLoss()
    input = torch.randn(1, 3, 288, 288)
    target = torch.randn(1, 3, 288, 288)
    loss = criterion(input, target, type='from_bottom_to_up')
    print(loss.item())
    loss = criterion(input, target, type='from_top_to_down')
    print(loss.item())
