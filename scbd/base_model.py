import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from likelihood import BernoulliNet, GaussianNet
from torch.optim import AdamW
from torchvision.models import resnet18, resnet50, densenet121
from utils.enum import Dataset, EncoderType


class SupConNet(nn.Module):
    def __init__(self, img_channels, encoder_type, z_size):
        super().__init__()
        if encoder_type == EncoderType.RESNET18:
            self.encoder = resnet18()
            self.r_size = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
            if img_channels > 3:
                self.encoder.conv1 = nn.Conv2d(img_channels, 64, 7, stride=2, padding=3, bias=False)
        elif encoder_type == EncoderType.RESNET50:
            self.encoder = resnet50()
            self.r_size = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
            if img_channels > 3:
                self.encoder.conv1 = nn.Conv2d(img_channels, 64, 7, stride=2, padding=3, bias=False)
        else:
            assert encoder_type == EncoderType.DENSENET121
            self.encoder = densenet121()
            self.r_size = self.encoder.classifier.in_features
            self.encoder.classifier = nn.Identity()
            if img_channels > 3:
                self.encoder.features[0] = nn.Conv2d(img_channels, 64, 7, stride=2, padding=3, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(self.r_size, self.r_size),
            nn.GELU(),
            nn.Linear(self.r_size, z_size)
        )

    def forward(self, x):
        r = self.encoder(x)
        z = F.normalize(self.proj(r), dim=1)
        return r, z


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class DCNN(nn.Module):
    def __init__(self, z_size, channels):
        super().__init__()
        layers = []
        layers.append(nn.Linear(z_size, (2 * channels) * (2 ** 2)))
        layers.append(nn.GELU())
        layers.append(nn.Unflatten(1, (-1, 2, 2)))
        layers.append(UpConv(2 * channels, 2 * channels)) # 2x2 -> 4x4
        layers.append(UpConv(2 * channels, 2 * channels)) # 4x4 -> 8x8
        layers.append(UpConv(2 * channels, 2 * channels)) # 8x8 -> 16x16
        layers.append(UpConv(2 * channels, channels))     # 16x16 -> 32x32
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class BaseModel(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.automatic_optimization = False
        self.net_c = SupConNet(self.hparams.img_channels, self.hparams.encoder_type, self.hparams.z_size)
        self.net_s = SupConNet(self.hparams.img_channels, self.hparams.encoder_type, self.hparams.z_size)
        self.dcnn_c = DCNN(self.hparams.z_size, self.hparams.channels)
        self.dcnn_s = DCNN(self.hparams.z_size, self.hparams.channels)
        if self.hparams.dataset == Dataset.CMNIST:
            self.likelihood = BernoulliNet(self.hparams.channels, self.hparams.img_channels)
        else:
            self.likelihood = GaussianNet(self.hparams.channels, self.hparams.img_channels)

    def get_inputs(self, batch):
        x, df = batch
        y = torch.tensor(df.y.values, device=self.device)
        e = torch.tensor(df.e.values, device=self.device)
        return x, y, e

    def get_supcon_loss(self, z, u):
        batch_size = len(z)
        u_col = u.unsqueeze(1)
        u_row = u.unsqueeze(0)
        mask = (u_col == u_row).float()
        offdiag_mask = 1. - torch.eye(batch_size, device=self.device)
        mask = mask * offdiag_mask
        logits = torch.matmul(z, z.T) / self.hparams.temperature
        p = mask / mask.sum(dim=1, keepdim=True).clamp(min=1.)
        q = F.log_softmax(logits, dim=1)
        cross_entropy = F.cross_entropy(q, p)
        return cross_entropy

    def get_invariance_loss(self, zc, e):
        batch_size = len(zc)
        e_col = e.unsqueeze(1)
        e_row = e.unsqueeze(0)
        mask_pos = (e_col == e_row).float()
        mask_neg = 1. - mask_pos
        offdiag_mask = 1. - torch.eye(batch_size, device=self.device)
        mask_pos = mask_pos * offdiag_mask
        logits = torch.matmul(zc, zc.T) / self.hparams.temperature
        q = F.log_softmax(logits, dim=1)
        log_prob_pos = (q * mask_pos).mean(dim=1)
        log_prob_neg = (q * mask_neg).mean(dim=1)
        return (log_prob_pos - log_prob_neg).abs().mean()

    def get_reconst_loss(self, x, zc, zs):
        x_pred = self.dcnn_c(zc.detach()) + self.dcnn_s(zs.detach())
        return -self.likelihood.logp(x_pred, x)

    def reconstruct(self, zc, zs):
        x_pred = self.dcnn_c(zc) + self.dcnn_s(zs)
        return self.likelihood.generate(x_pred)

    def forward(self, x, y, s, e):
        rc, zc = self.net_c(x)
        _, zs = self.net_s(x)
        supcon_loss_c = self.get_supcon_loss(zc, y)
        supcon_loss_s = self.get_supcon_loss(zs, s)
        invariance_loss = self.get_invariance_loss(zc, e)
        reconst_loss = self.get_reconst_loss(x, zc, zs)
        return supcon_loss_c, supcon_loss_s, invariance_loss, reconst_loss, zc, zs, rc

    def configure_optimizers(self):
        params_main, params_aux = [], []
        params_main += list(self.net_c.parameters())
        params_main += list(self.net_s.parameters())
        opt_main = AdamW(params_main, lr=self.hparams.lr, weight_decay=self.hparams.wd)
        params_aux += list(self.dcnn_c.parameters())
        params_aux += list(self.dcnn_s.parameters())
        params_aux += list(self.likelihood.parameters())
        return opt_main, params_aux