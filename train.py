from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
import pytorch_lightning as pl

from ical.datamodule import HMEDatamodule
from ical.lit_ical import LitICAL

import argparse
from sconf import Config
from pytorch_lightning.loggers import WandbLogger as Logger


def train(config: Config):
    model_module = LitICAL(
        d_model=config.model.d_model,
        # encoder
        growth_rate=config.model.growth_rate,
        num_layers=config.model.num_layers,
        # decoder
        nhead=config.model.nhead,
        num_decoder_layers=config.model.num_decoder_layers,
        dim_feedforward=config.model.dim_feedforward,
        dc=config.model.dc,
        dropout=config.model.dropout,
        vocab_size=config.model.vocab_size,
        cross_coverage=config.model.cross_coverage,
        self_coverage=config.model.self_coverage,
        # beam search
        beam_size=config.model.beam_size,
        max_len=config.model.max_len,
        alpha=config.model.alpha,
        early_stopping=config.model.early_stopping,
        temperature=config.model.temperature,
        # training
        learning_rate=config.model.learning_rate,
        patience=config.model.patience,
        dynamic_weight=config.model.dynamic_weight,
    )
    data_module = HMEDatamodule(
        folder=config.data.folder,
        test_folder=config.data.test_folder,
        max_size=config.data.max_size,
        scale_to_limit=config.data.scale_to_limit,
        train_batch_size=config.data.train_batch_size,
        eval_batch_size=config.data.eval_batch_size,
        num_workers=config.data.num_workers,
        scale_aug=config.data.scale_aug,
    )

    # logger = Logger(name=config.wandb.name,
    #                 project=config.wandb.project,
    #                 log_model=config.wandb.log_model,
    #                 config=dict(config),
    #                 )
    # logger.watch(model_module,
    #              log="all",
    #              log_freq=100
    #              )
    lr_callback = pl.callbacks.LearningRateMonitor(
        logging_interval=config.trainer.callbacks[0].init_args.logging_interval)

    lasted_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoint",
        filename="lasted",
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        monitor=None,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="lightning_logs",
        save_top_k=config.trainer.callbacks[1].init_args.save_top_k,
        monitor=config.trainer.callbacks[1].init_args.monitor,
        mode=config.trainer.callbacks[1].init_args.mode,
        filename=config.trainer.callbacks[1].init_args.filename)

    trainer = pl.Trainer(
        gpus=config.trainer.gpus,
        accelerator=config.trainer.accelerator,
        max_epochs=config.trainer.max_epochs,
        deterministic=config.trainer.deterministic,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,

        plugins=DDPPlugin(find_unused_parameters=False),
        # logger=logger,
        callbacks=[lr_callback, checkpoint_callback, lasted_checkpoint_callback],
    )

    trainer.fit(model_module, data_module)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = Config(args.config)
    train(config)

