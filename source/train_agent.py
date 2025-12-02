from os import name
import random
from pathlib import Path
from typing import List

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from argparser import get_args
from catch_module import CatchRLModule
from video_logger import VideoLoggerCallback

PROJECT_NAME = "RL-Catch"
LOGS_DIR = Path("logs/")


def train(hparams, config=None):
    logger = WandbLogger(name=f"{hparams.run_name}_{hparams.algorithm}",
                         project=PROJECT_NAME,
                         save_dir=LOGS_DIR,
                         log_model=True,
                         anonymous="allow",)
    csv_logger = CSVLogger(save_dir=LOGS_DIR)

    hparams: dict = vars(hparams)
    ckpt_path = hparams.pop("ckpt_path")
    start_from_scratch_with_ckpt = hparams.pop("start_from_scratch_with_ckpt")
    reinit_last_layer = hparams.pop("reinit_last_layer")
    hparams.pop("periodic_resetting")
    hparams.pop('run_name')
    max_epochs = hparams.pop('max_epochs')
    log_video = hparams.pop('log_video')

    callbacks = []
    if log_video:
        callbacks.append(VideoLoggerCallback())

    hparams = {**hparams, **config} if config else hparams

    catch_module = CatchRLModule(**hparams)

    if ckpt_path and start_from_scratch_with_ckpt:
        print(f"Loading model weights from {ckpt_path} for transfer learning, but starting training from scratch.")
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        catch_module.load_state_dict(checkpoint["state_dict"])

        if reinit_last_layer:
            catch_module.reinit_last_layer()

        ckpt_path = None  # Ensure trainer.fit starts from scratch

    trainer = Trainer(max_epochs=max_epochs,
                      logger=[logger, csv_logger],
                      log_every_n_steps=1,
                      callbacks=callbacks,
                      )
    trainer.fit(catch_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    hparams = get_args()
    train(hparams)
