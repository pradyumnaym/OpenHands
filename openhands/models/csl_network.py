import torch
import torch.nn as nn
import pytorch_lightning as pl

class CSLRModel(pl.LightningModule):
  def __init__(self, encoder_seq, config, num_classes_gloss):
    super().__init__()
    self.config = config
    self.encoder_seq = encoder_seq
    self.num_encoders = len(encoder_seq)
    self.num_classes_gloss = num_classes_gloss
    self.classifiers = nn.ModuleDict() ## to use features of an encoder to classify
    for enc in self.encoder_seq:
      if enc.use_ctc:
        self.classifiers[f'{enc.encoder_id}'] = nn.Linear(enc.out_size, self.num_classes_gloss)
    self.externals = {}
    self.initialize_losses(config.losses)

  def initialize_losses(self, loss_config):
    self.loss_config = loss_config
    self.loss_fn = {} # stores class objects to calculate losses
    self.loss_ip = {} # stores name of inputs to losses
    self.loss_value = {} # stores loss values for a given iter
    
    self.loss_fn['CTC'] = nn.CTCLoss(reduction='none', zero_infinity=False) ## CTC loss will always be used
    i=1
    while True:
      if not hasattr(self.loss_config, f'loss{i}'):
        break
      loss_i = getattr(self.loss_config, f'loss{i}')
      
      if loss_i.type == 'Distillation':
        from .continuous.loss.tlp import SeqKD
        self.loss_fn[loss_i.name] = SeqKD(**loss_i.params)
      # elif loss_type == 'CTC':
      #   self.loss_fn[loss_name] = nn.CTCLoss(reduction='none', zero_infinity=False)
      
      self.loss_ip[loss_i.name] = loss_i.inputs
      i+=1
      

  def forward(self, x, len_x, label, len_label, is_training=True):
    for i, enc in enumerate(self.encoder_seq):
      x, len_x, internal_losses = enc(x,len_x)
      self.loss_value.update(internal_losses)
      
      if enc.use_ctc:
        logits = self.classifiers[f'{enc.encoder_id}'](x)
        self.externals[f'encoder{i+1}.logits'] = logits
        self.loss_value[f'{enc.encoder_id}.CTCLoss'] = self.loss_fn['CTC'](
                                                                logits.transpose(0,1).log_softmax(-1), 
                                                                label.cpu().int(), 
                                                                len_x.cpu().int(), 
                                                                len_label.cpu().int()).mean()
        ## get predictions from output of last encoder
        # if enc.encoder_id == f'encoder{self.num_encoders}':
        #   pred = self.decoder.decode(logits, len_x, batch_first=False, probs=False)
    return self.compute_external_losses()
    
  
  def compute_external_losses(self):
    for loss_name, loss_fn in self.loss_fn.items():
      if loss_name == 'CTC':
        continue
      
      ip_args = {}
      for ip_arg, ext_var in self.loss_ip[loss_name].items():
        ip_args[ip_arg] = self.externals[ext_var]
      
      self.loss_value[loss_name] = loss_fn(**ip_args)
    
    total_loss = 0.0
    for loss_name, loss_value in self.loss_value.items():
      total_loss += (getattr(self.config.loss_weights, loss_name) * loss_value)
    return total_loss, self.loss_value


  def training_step(self, batch, batch_idx):
    # x_img = torch.randn((5,84,3,224,224))
    # x_len = torch.tensor([84,80,76,68,62], dtype=torch.int)
    # label = torch.randint(0,1296,(29,))
    # label_len = torch.tensor([4,8,7,6,4], dtype=torch.int)
    
    loss, _ = self.forward(batch['frames'], batch['frames_len'], batch['gloss'], batch['gloss_len'])
    return loss


  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.config.optimizer_args.base_lr)


# class G2TModel(pl.LightningModule):
#   def __init__(self, config_encoder, config_decoder, custom_classes='False'):
#     super().__init__()
#     if custom_classes:


# class SLTModel(pl.LightningModule):
#   def __init__(self, s2g_config, g2t_config, model_type='s2g2t'):
#     super().__init__()
#     self.s2g_model = CSLRModel(s2g_config)
#     self.g2t_model = G2TModel(g2t_config['encoder'], g2t_config['decoder'], custom_classes=g2t_config['custom_class'])