import torch.nn as nn


class CSLRModel(nn.Module):
  def __init__(self, encoder_seq, loss_config, num_classes_gloss):
    super().__init__()
    self.encoder_seq = encoder_seq
    self.num_encoders = len(encoder_seq)
    self.num_classes_gloss = num_classes_gloss
    self.classifiers = nn.ModuleDict() ## to use features of an encoder to classify
    for enc in self.encoder_seq:
      if enc.use_ctc:
        self.classifiers[f'{enc.encoder_id}'] = nn.Linear(enc.out_size, self.num_classes_gloss)
    self.externals = {}
    self.initialize_losses(loss_config)

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
        self.externals[f'encoder{i}.logits'] = logits
        self.loss_value[f'{enc.encoder_id}.CTCLoss'] = self.loss_fn['CTC'](
                                                                logits.log_softmax(-1), 
                                                                label.cpu().int(), 
                                                                len_x.cpu().int(), 
                                                                len_label.cpu().int()).mean()
        ## get predictions from output of last encoder
        # if enc.encoder_id == f'encoder{self.num_encoders}':
        #   pred = self.decoder.decode(logits, len_x, batch_first=False, probs=False)
      
      return self.compute_external_losses()
    
  
  def compute_external_losses(self):
    for loss_name, loss_fn in self.loss_fn.items():
      self.loss_value[loss_name] = loss_fn(**self.loss_ip[loss_name])
    
    total_loss = 0.0
    for loss_name, loss_value in self.loss_value.items():
      total_loss += (getattr(self.loss_config.loss_weights, loss_name) * loss_value)
    return total_loss, self.loss_value


