import torch
import torch.nn as nn
import torch.nn.functional as F


class DividedSsimLoss(nn.Module):
    def __init__(self, SSIM_c1=0.2, k_Loss=[9, 8, 7, 6, 5, 4, 3, 2, 1]):
        super(DividedSsimLoss, self).__init__()
        self.SSIM_c1 = SSIM_c1
        self.k_Loss = k_Loss
        self.rgb2gray = torch.FloatTensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        self.min_size = 256

    def SSIM(self, input, target):
        mu_x = input.sum(dim=[1,2])
        mu_y = target.sum(dim=[1,2])
        SSIM_L = (2. * mu_x * mu_y + self.SSIM_c1) / (mu_x**2 + mu_y**2 + self.SSIM_c1)
        return SSIM_L.mean()

    def SSIM_Loss(self, input, target):
        return 1 - self.SSIM(input, target)

    def calc_dfs(self, input, target, deep):
        ret = self.SSIM_Loss(input, target) * self.k_Loss[deep]
        B, H, W = input.shape
        if H == 1 or W == 1:
            return ret
        mid = (H + 1) // 2
        ret += 0.25 * (
                self.calc_dfs(input[:, :mid, :mid], target[:, :mid, :mid], deep + 1) +
                self.calc_dfs(input[:, :mid, mid:], target[:, :mid, mid:], deep + 1) +
                self.calc_dfs(input[:, mid:, :mid], target[:, mid:, :mid], deep + 1) +
                self.calc_dfs(input[:, mid:, mid:], target[:, mid:, mid:], deep + 1)
        )
        return ret

    def forward(self, input, target):
        assert(input.shape == target.shape)
        B, C, H, W = input.shape
        L = self.min_size
        assert(H >= L and W >= L)

        pad_h = -H % L
        pad_w = -W % L
        input_pad = F.pad(input, [0, pad_h, 0, pad_w], mode='reflect')
        target_pad = F.pad(target, [0, pad_h, 0, pad_w], mode='reflect')

        if C == 3:
            device = torch.get_device(input) if 'cuda' in input.type() else 'cpu'
            self.rgb2gray = self.rgb2gray.to(device)
            input_gray = (input_pad*self.rgb2gray).sum(dim=1).squeeze(1)
            target_gray = (target_pad*self.rgb2gray).sum(dim=1).squeeze(1)
        else:
            input_gray = input_pad.squeeze(1)
            target_gray = target_pad.squeeze(1)

        B, H, W = input_gray.shape
        loss_tile = []
        for h in range(0, H, L):
            for w in range(0, W, L):
                print(1)
                loss_tile.append(self.calc_dfs(input_gray[:, h:h+L, w:w+L], target_gray[:, h:h+L, w:w+L], 0))
                print(2)
        return sum(loss_tile) / len(loss_tile)


if __name__ == "__main__":
    criterion = DividedSsimLoss()
    input = torch.randn(4, 3, 288, 288).cuda()
    target = torch.randn(4, 3, 288, 288).cuda()
    loss = criterion(input, target)
    print(loss.item())
