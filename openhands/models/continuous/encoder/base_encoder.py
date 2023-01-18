import torch
import torch.nn.functional as F
from torchvision import models

class EncoderModule(torch.nn.Module):
  def __init__(self, encoder_id="__", use_ctc=False, **kwargs):
    super(EncoderModule, self).__init__()
    self.use_ctc = use_ctc
    self.encoder_id = encoder_id
    self.internal_losses = {}

class ResNet(EncoderModule):
  def __init__(self, **kwargs):
    super(ResNet, self).__init__(**kwargs)
    resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    self.conv = torch.nn.Sequential(*(list(resnet_model.children())[:-1])) #remove the last FC layer

  def forward(self, x, batch):

    B, T, C, H, W = x.size()
    x = x.reshape(B * T, C, H, W)
    x = self.conv(x)
    x = x.reshape(B, T, -1)
    return x
