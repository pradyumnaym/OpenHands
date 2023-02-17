import omegaconf
from openhands.apis.classification_model import ClassificationModel
from openhands.core.exp_utils import get_trainer
from pytorch_lightning import seed_everything

seed_everything(2023, workers=True)

cfg = omegaconf.OmegaConf.load("examples/configs/wlasl/decoupled_gcn.yaml")
trainer = get_trainer(cfg)

model = ClassificationModel(cfg=cfg, trainer=trainer)
#model.init_from_checkpoint_if_available()
model.fit()
