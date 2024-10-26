import os
import torch.nn as nn
import torch.nn.functional as F
from base_model import BaseModel
from torch.optim import AdamW
from torchmetrics import Accuracy
from utils.nn_utils import shuffle_batch
from utils.plot import *


class DomainGeneralization(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.save_hyperparameters(hparams)
        self.classifier_y_rc = nn.Linear(self.net_c.r_size, self.hparams.y_size)
        self.classifier_e_rc = nn.Linear(self.net_c.r_size, self.hparams.e_size)
        self.val_acc_y_rc = Accuracy("multiclass", num_classes=int(self.hparams.y_size))
        self.val_acc_e_rc = Accuracy("multiclass", num_classes=int(self.hparams.e_size))
        self.test_acc_y_rc = Accuracy("multiclass", num_classes=int(self.hparams.y_size))

    def get_ye(self, y, e):
        return y * self.hparams.e_size + e

    def plot(self, x, zc, zs, stage):
        x_reconst = self.reconstruct(zc, zs)
        x_reconst_shuffle_zc = self.reconstruct(shuffle_batch(zc), zs)
        x_reconst_shuffle_zs = self.reconstruct(zc, shuffle_batch(zs))
        fig, axes = plt.subplots(4, N_COLS, figsize=(N_COLS, 4))
        for ax in axes.flatten():
            remove_ticks(ax)
        for col_idx in range(N_COLS):
            plot_image(axes[0, col_idx], x[col_idx])
            plot_image(axes[1, col_idx], x_reconst[col_idx])
            plot_image(axes[2, col_idx], x_reconst_shuffle_zc[col_idx])
            plot_image(axes[3, col_idx], x_reconst_shuffle_zs[col_idx])
        dpath = os.path.join(self.trainer.log_dir, "fig", stage)
        os.makedirs(dpath, exist_ok=True)
        plt.savefig(os.path.join(dpath, f"step={self.global_step}.png"))
        plt.close()

    def training_step(self, batch, batch_idx):
        opt_main, opt_aux = self.optimizers()
        x, y, e = self.get_inputs(batch)
        ye = self.get_ye(y, e)
        supcon_loss_c, supcon_loss_s, invariance_loss, reconst_loss, zc, zs, rc = self(x, y, ye, e)
        # Main
        self.toggle_optimizer(opt_main)
        loss_main = supcon_loss_c + supcon_loss_s + self.hparams.alpha * invariance_loss
        self.manual_backward(loss_main)
        opt_main.step()
        opt_main.zero_grad()
        self.untoggle_optimizer(opt_main)
        # Auxiliary
        self.toggle_optimizer(opt_aux)
        loss_y_rc = F.cross_entropy(self.classifier_y_rc(rc.detach()), y)
        loss_e_rc = F.cross_entropy(self.classifier_e_rc(rc.detach()), e)
        loss_aux = reconst_loss + loss_y_rc + loss_e_rc
        self.manual_backward(loss_aux)
        opt_aux.step()
        opt_aux.zero_grad()
        self.untoggle_optimizer(opt_aux)
        if self.global_step > 0 and self.global_step % self.hparams.steps_per_plot == 0 and self.trainer.is_global_zero:
            self.plot(x, zc, zs, "train")

    def validation_step(self, batch, batch_idx):
        x, y, e = self.get_inputs(batch)
        ye = self.get_ye(y, e)
        supcon_loss_c, supcon_loss_s, invariance_loss, reconst_loss, zc, zs, rc = self(x, y, ye, e)
        loss = supcon_loss_c + supcon_loss_s + self.hparams.alpha * invariance_loss
        self.log("val_supcon_loss_c", supcon_loss_c, on_step=False, on_epoch=True)
        self.log("val_supcon_loss_s", supcon_loss_s, on_step=False, on_epoch=True)
        self.log("val_invariance_loss", invariance_loss, on_step=False, on_epoch=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_acc_y_rc(self.classifier_y_rc(rc), y)
        self.val_acc_e_rc(self.classifier_e_rc(rc), e)
        self.log("val_reconst_loss", reconst_loss, on_step=False, on_epoch=True)
        self.log("val_acc_y_rc", self.val_acc_y_rc, on_step=False, on_epoch=True)
        self.log("val_acc_e_rc", self.val_acc_e_rc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y, e = self.get_inputs(batch)
        rc, zc = self.net_c(x)
        self.test_acc_y_rc(self.classifier_y_rc(rc), y)
        self.log("test_acc_y_rc", self.test_acc_y_rc, on_step=False, on_epoch=True)
        if batch_idx == 0:
            _, zs = self.net_s(x)
            self.plot(x, zc, zs, "test")

    def configure_optimizers(self):
        opt_main, params_aux = super().configure_optimizers()
        params_aux += list(self.classifier_y_rc.parameters())
        params_aux += list(self.classifier_e_rc.parameters())
        opt_aux = AdamW(params_aux, lr=self.hparams.lr, weight_decay=self.hparams.wd)
        return opt_main, opt_aux