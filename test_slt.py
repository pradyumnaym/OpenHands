import omegaconf
import pytorch_lightning as pl

from pytorch_lightning.loggers.wandb import WandbLogger
from openhands.models.csl_network import SLTModel
from pytorch_lightning.callbacks import LearningRateMonitor

if __name__ == '__main__':
  config = omegaconf.OmegaConf.load("examples/configs/phoenix14-t/slt_features.yaml")
  wandb_logger = WandbLogger(project='slt')
  lr_monitor = LearningRateMonitor(logging_interval='step')
  #trainer = pl.Trainer(**config.trainer, logger=wandb_logger, callbacks=[lr_monitor])
  trainer = pl.Trainer(**config.trainer, callbacks=[lr_monitor])
  model = SLTModel(config, trainer)
  model.fit()
