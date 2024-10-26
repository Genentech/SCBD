import boto3
import gc
import numpy as np
import os
import pandas as pd
import pickle
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO
from torch.optim import AdamW
from torchmetrics import F1Score
from base_model import BaseModel


class BatchCorrection(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.save_hyperparameters(hparams)
        self.classifier_e_zc = nn.Linear(self.hparams.z_size, self.hparams.e_size)
        self.classifier_e_zs = nn.Linear(self.hparams.z_size, self.hparams.e_size)
        self.val_f1_score_e_zc = F1Score("multiclass", num_classes=int(self.hparams.e_size))
        self.val_f1_score_e_zs = F1Score("multiclass", num_classes=int(self.hparams.e_size))
        self.s3_client = boto3.client("s3")

    def training_step(self, batch, batch_idx):
        opt_main, opt_aux = self.optimizers()
        x, y, e = self.get_inputs(batch)
        supcon_loss_c, supcon_loss_s, invariance_loss, reconst_loss, zc, zs, _ = self(x, y, e, e)
        # Main
        self.toggle_optimizer(opt_main)
        loss_main = supcon_loss_c + supcon_loss_s + self.hparams.alpha * invariance_loss
        self.manual_backward(loss_main)
        opt_main.step()
        opt_main.zero_grad()
        self.untoggle_optimizer(opt_main)
        # Auxiliary
        self.toggle_optimizer(opt_aux)
        loss_e_zc = F.cross_entropy(self.classifier_e_zc(zc.detach()), e)
        loss_e_zs = F.cross_entropy(self.classifier_e_zs(zs.detach()), e)
        loss_aux = reconst_loss + loss_e_zc + loss_e_zs
        self.manual_backward(loss_aux)
        opt_aux.step()
        opt_aux.zero_grad()
        self.untoggle_optimizer(opt_aux)

    def validation_step(self, batch, batch_idx):
        x, y, e = self.get_inputs(batch)
        supcon_loss_c, supcon_loss_s, invariance_loss, reconst_loss, zc, zs, _ = self(x, y, e, e)
        loss = supcon_loss_c + supcon_loss_s + self.hparams.alpha * invariance_loss
        self.log("val_supcon_loss_c", supcon_loss_c, on_step=False, on_epoch=True)
        self.log("val_supcon_loss_s", supcon_loss_s, on_step=False, on_epoch=True)
        self.log("val_invariance_loss", invariance_loss, on_step=False, on_epoch=True)
        self.log("val_reconst_loss", reconst_loss, on_step=False, on_epoch=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        e_pred_zc = self.classifier_e_zc(zc)
        e_pred_zs = self.classifier_e_zs(zs)
        self.val_f1_score_e_zc(e_pred_zc, e)
        self.val_f1_score_e_zs(e_pred_zs, e)
        self.log("val_f1_score_e_zc", self.val_f1_score_e_zc, on_step=False, on_epoch=True)
        self.log("val_f1_score_e_zs", self.val_f1_score_e_zs, on_step=False, on_epoch=True)

    def save_embed(self):
        df = pd.concat(self.df)
        zc = np.concatenate(self.zc)
        zs = np.concatenate(self.zs)
        buffer = BytesIO()
        pickle.dump((df, zc, zs), buffer)
        buffer.seek(0)
        fpath = os.path.join(self.hparams.results_dpath, f"version_{self.hparams.seed}", f"embed_{self.test_step_count}.pkl")
        self.s3_client.upload_fileobj(buffer, self.hparams.s3_bucket_name, fpath)
        self.df.clear()
        self.zc.clear()
        self.zs.clear()
        del self.df
        del self.zc
        del self.zs
        gc.collect()
        self.df, self.zc, self.zs = [], [], []

    def on_test_start(self):
        self.df, self.zc, self.zs = [], [], []
        self.test_step_count = 0

    def test_step(self, batch, batch_idx):
        x, df = batch
        x, y, e = self.get_inputs(batch)
        _, zc = self.net_c(x)
        _, zs = self.net_s(x)
        self.df.append(df)
        self.zc.append(zc.cpu().numpy())
        self.zs.append(zs.cpu().numpy())
        if self.test_step_count > 0 and self.test_step_count % self.hparams.steps_per_embed == 0:
            self.save_embed()
        self.test_step_count += 1

    def on_test_end(self):
        self.save_embed()

    def configure_optimizers(self):
        opt_main, params_aux = super().configure_optimizers()
        params_aux += list(self.classifier_e_zc.parameters())
        params_aux += list(self.classifier_e_zs.parameters())
        opt_aux = AdamW(params_aux, lr=self.hparams.lr, weight_decay=self.hparams.wd)
        return opt_main, opt_aux