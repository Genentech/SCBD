import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class BernoulliNet(nn.Module):
    def __init__(self, in_channels, img_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, img_channels, 1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))

    def logp(self, x_pred, x):
        return -F.binary_cross_entropy(self(x_pred), x)

    def generate(self, x):
        return self(x)


class GaussianNet(nn.Module):
    def __init__(self, in_channels, img_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, img_channels, 1)
        self.logsd = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x_pred):
        return self.conv(x_pred)

    def logp(self, x_pred, x):
        return Normal(self(x_pred), self.logsd.exp()).log_prob(x).mean()

    def generate(self, x_pred):
        return self(x_pred).clip(0., 1.)