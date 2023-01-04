import omegaconf
import pytorch_lightning as pl

from pytorch_lightning.loggers.wandb import WandbLogger
from openhands.models.csl_network import SLTModel

if __name__ == '__main__':
  config = omegaconf.OmegaConf.load("examples/configs/phoenix14-t/slt.yaml")
  wandb_logger = WandbLogger(project='slt')
  trainer = pl.Trainer(**config.trainer, logger=wandb_logger)
  model = SLTModel(config, trainer)
  model.fit()
