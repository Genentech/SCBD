import data.camelyon17
import data.cmnist
import data.funk22
import data.rxrx1
import lightning as L
from argparse import ArgumentParser
from batch_correction import BatchCorrection
from domain_generalization import DomainGeneralization
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from utils.enum import Task, Dataset, DataSplit, EncoderType, EType
from utils.nn_utils import HParams


def main(hparams):
    L.seed_everything(hparams.seed)
    if hparams.dataset == Dataset.CAMELYON17:
        data_module = data.camelyon17
        task = Task.DOMAIN_GENERALIZATION
    elif hparams.dataset == Dataset.CMNIST:
        data_module = data.cmnist
        task = Task.DOMAIN_GENERALIZATION
    elif hparams.dataset == Dataset.FUNK22:
        data_module = data.funk22
        task = Task.BATCH_CORRECTION
    else:
        assert hparams.dataset == Dataset.RXRX1
        data_module = data.rxrx1
        task = Task.DOMAIN_GENERALIZATION
    data_train, data_val, data_test, metadata = data_module.get_data(hparams)
    hparams.img_channels = metadata["img_channels"]
    hparams.y_size = metadata["y_size"]
    hparams.e_size = metadata["e_size"]
    logger = CSVLogger(hparams.results_dpath, name="", version=hparams.seed)
    if task == Task.BATCH_CORRECTION:
        assert hparams.s3_bucket_name is not None
        if hparams.ckpt_fpath is None:
            model = BatchCorrection(hparams)
        else:
            model = BatchCorrection.load_from_checkpoint(hparams.ckpt_fpath, results_dpath=hparams.results_dpath)
        trainer = L.Trainer(
            logger=logger,
            callbacks=[ModelCheckpoint(filename="{step}", save_top_k=-1)],
            max_steps=hparams.steps,
            val_check_interval=hparams.steps_per_val,
            limit_val_batches=hparams.limit_val_batches,
            deterministic=True
        )
        if hparams.data_split is None:
            trainer.fit(model, data_train, data_val, ckpt_path=hparams.ckpt_fpath)
        elif hparams.data_split == DataSplit.TRAIN:
            trainer.test(model, data_train)
        elif hparams.data_split == DataSplit.VAL:
            trainer.test(model, data_val)
        else:
            assert hparams.data_split == DataSplit.TEST
            trainer.test(model, data_test)
    else:
        assert task ==Task.DOMAIN_GENERALIZATION
        if hparams.ckpt_fpath is None:
            model = DomainGeneralization(hparams)
        else:
            model = DomainGeneralization.load_from_checkpoint(hparams.ckpt_fpath)
        ckpt = ModelCheckpoint(filename="best", monitor="val_loss")
        steps_per_epoch = len(data_train)
        if hparams.steps_per_val < steps_per_epoch:
            trainer = L.Trainer(
                logger=logger,
                callbacks=[ckpt],
                max_epochs=-1,
                max_steps=hparams.steps,
                val_check_interval=hparams.steps_per_val,
                deterministic=True
            )
        else:
            check_val_every_n_epoch = hparams.steps_per_val // steps_per_epoch
            trainer = L.Trainer(
                logger=logger,
                callbacks=[ckpt],
                max_epochs=-1,
                max_steps=hparams.steps,
                check_val_every_n_epoch=check_val_every_n_epoch,
                deterministic=True
            )
        trainer.fit(model, data_train, data_val, ckpt_path=hparams.ckpt_fpath)
        trainer.test(model, data_test, ckpt_path="best")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Dataset, choices=list(Dataset), required=True)
    parser.add_argument("--data_split", type=DataSplit, choices=list(DataSplit))
    parser.add_argument("--data_dpath", type=str, required=True)
    parser.add_argument("--results_dpath", type=str, required=True)
    parser.add_argument("--s3_bucket_name", type=str)
    parser.add_argument("--ckpt_fpath", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--e_type", type=EType, choices=list(EType), default=EType.PLATE_WELL)
    # Model
    parser.add_argument("--encoder_type", type=EncoderType, choices=list(EncoderType), required=True)
    parser.add_argument("--z_size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.)
    # Optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--y_per_batch", type=int, default=128)
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--steps_per_plot", type=int, default=5000)
    parser.add_argument("--steps_per_val", type=int, default=5000)
    parser.add_argument("--limit_val_batches", type=int, default=500)
    parser.add_argument("--steps_per_embed", type=int, default=1000)
    hparams = HParams()
    hparams.update(parser.parse_args().__dict__)
    main(hparams)