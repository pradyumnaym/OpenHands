#import torch
import omegaconf
from openhands.core import DataModuleContinuous
from openhands.models.csl_loader import get_cslr_model
import pytorch_lightning as pl

## Hardcoded inputs
# x_img = torch.randn((5,84,3,224,224))
# x_len = torch.tensor([84,80,76,68,62], dtype=torch.int)
# label = torch.randint(0,1296,(29,))
# label_len = torch.tensor([4,8,7,6,4], dtype=torch.int)

if __name__ == '__main__':
  config = omegaconf.OmegaConf.load("examples/configs/phoenix14/tlp.yaml")

  datamodule = DataModuleContinuous(config.data)
  
  model = get_cslr_model(config.model)

  trainer = pl.Trainer(**config.trainer)
  trainer.fit(model, datamodule=datamodule)
  
