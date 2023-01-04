import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import pytorch_lightning as pl

from openhands.datasets.continuous.vocabulary import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
from .continuous.loss.slt import XentLoss
from .continuous.decoder.decoder_utils import ctc_decode, decode
from .continuous.metrics.metrics import get_wer, get_bleu
from ..core.data import DataModule
from .csl_loader import get_slt_model


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


class SLTModel(pl.LightningModule):
  def __init__(self, config, trainer):
    super().__init__()

    self.config = config
    self.trainer = trainer
    self.datamodule = DataModule(config.data)
    self.datamodule.setup(stage='fit')
    self.train_dataloader = self.datamodule.train_dataloader()
    self.val_dataloader = self.datamodule.val_dataloader()

    config.model.data_info.num_classes_gloss = len(self.datamodule.gloss_vocab)
    config.model.data_info.num_classes_text = len(self.datamodule.text_vocab)
    config.model.decoder.params.data_info = config.model.data_info
    self.num_classes_gloss = len(self.datamodule.gloss_vocab)
    self.num_classes_text = len(self.datamodule.text_vocab)
    self.pad_index_gloss = self.datamodule.gloss_vocab[PAD_TOKEN]
    self.pad_index_text = self.datamodule.text_vocab[PAD_TOKEN]
    self.bos_index_text = self.datamodule.text_vocab[BOS_TOKEN]
    self.eos_index_text = self.datamodule.text_vocab[EOS_TOKEN]

    self.encoder_seq, self.decoder = get_slt_model(config.model)
    self.num_encoders = len(self.encoder_seq)
    self.gloss_output_layer = nn.Linear(self.encoder_seq[-1].out_size, self.num_classes_gloss)

    self.externals = {}
    self.initialize_losses(config.model.losses)
    self.save_hyperparameters()

  def initialize_losses(self, loss_config):
    self.loss_config = loss_config
    self.loss_fn = {} # stores class objects to calculate losses
    self.loss_ip = {} # stores name of inputs to losses
    self.loss_value = {} # stores loss values for a given iter
    
    self.loss_fn['CTC'] = nn.CTCLoss(zero_infinity=False) ## CTC loss will always be used
    self.loss_fn['Xent'] = XentLoss(pad_index=self.pad_index_text, smoothing=0.0)


  def forward(self, batch):
    for i, enc in enumerate(self.encoder_seq):
      if i==0:
        x = enc(batch)
      else:
        x = enc(x, batch)

      if i == len(self.encoder_seq) - 1:
        # this is the last encoder, perform CSLR here
        encoder_output = x
        gloss_scores = self.gloss_output_layer(x)
        gloss_probabilities = gloss_scores.log_softmax(2)
        gloss_probabilities = gloss_probabilities.permute(1, 0, 2) # Both torch and tf CTC functions require (T, N, C)

    word_scores, *_ = self.decoder(x, batch)
    word_probabilities = word_scores.log_softmax(-1)

    return gloss_probabilities, word_probabilities, encoder_output

  def training_step(self, batch, batch_idx):
    gloss_probabilities, word_probabilities, *_ = self.forward(batch)
    self.loss_value['CTC'] = self.loss_fn['CTC'](
      gloss_probabilities,
      batch["gloss"],
      batch["frames_len"],
      batch["gloss_len"]
    ) * getattr(self.config.model.loss_weights, 'CTC')

    self.loss_value['Xent'] = self.loss_fn["Xent"](
      word_probabilities, batch["text"]
    ) * getattr(self.config.model.loss_weights, "Xent")

    final_loss = sum(self.loss_value.values())
    for loss, value in self.loss_value.items():
      self.log(loss, value, on_step=True, on_epoch=True)
    self.log("train_loss", final_loss, on_step=True, on_epoch=True)
    return final_loss

  def validation_step(self, batch, batch_idx):
    gloss_probabilities, word_probabilities, encoder_output = self.forward(batch)
    decoded_gloss_sequences = ctc_decode(
      gloss_probabilities,
      batch["frames_len"],
      self.config.model.recognition_beam_size,

    )

    stacked_text_output, _ = decode(
      decoder=self.decoder,
      translation_beam_size=self.config.model.translation_beam_size,
      translation_max_output_length=self.config.model.translation_max_output_length,
      translation_beam_alpha=self.config.model.eval_translation_beam_alpha,
      encoder_output=encoder_output,
      frames_mask=batch['frames_mask'],
      bos_index=self.bos_index_text,
      eos_index=self.eos_index_text,
      pad_index=self.pad_index_text
    )

    metrics_dict = get_wer(
      cleanup_function=self.train_dataloader.dataset.clean_glosses,
      decoded_gloss_sequences=decoded_gloss_sequences,
      reference_gloss_sequences=batch["raw_gloss"],
      gloss_vocab=self.datamodule.gloss_vocab
    )

    metrics_dict.update(
      get_bleu(
        decoded_text_sequences=stacked_text_output,
        reference_text_sequences=batch["raw_text"],
        text_vocab=self.datamodule.text_vocab
      )
    )
    
    for metric, value in metrics_dict.items():
      if 'wer' in metric or 'bleu4' in metric:
        self.log(metric, value, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)
      else:
        self.log(metric, value, sync_dist=True, on_step=False, on_epoch=True)

    return {
      "gloss_probabilities": gloss_probabilities,
      "word_probabilities": word_probabilities
    }

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.config.model.optimizer_args.base_lr)

  def fit(self):
     self.trainer.fit(
      self,
      train_dataloaders=self.train_dataloader,
      val_dataloaders=self.val_dataloader,
    )
