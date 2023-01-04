import omegaconf
import torch.nn as nn
import hydra

def load_encoder(encoder_cfg, enc_id):
  if encoder_cfg.type == "custom":
    return hydra.utils.instantiate(**encoder_cfg.params)
  
  if encoder_cfg.type == "TLPResNet18":
    from .continuous.encoder.tlp import TLPResnet18
    return TLPResnet18(encoder_id=enc_id)
  
  if encoder_cfg.type == "TempConv_TLP":
    from .continuous.encoder.tlp import TemporalConv
    return TemporalConv(encoder_id=enc_id, **encoder_cfg.params)
  
  if encoder_cfg.type == "TempConv_VAC":
    from .continuous.encoder.vac import TemporalConv
    return TemporalConv(encoder_id=enc_id, **encoder_cfg.params)
  
  if encoder_cfg.type == "BiLSTMLayer":
    from .continuous.encoder.tlp import BiLSTMLayer
    return BiLSTMLayer(encoder_id=enc_id, **encoder_cfg.params)

  if encoder_cfg.type == "ResNet":
    from .continuous.encoder.base_encoder import ResNet
    return ResNet(encoder_id=enc_id)

  if encoder_cfg.type == "TransformerEncoder":
    from .continuous.encoder.transformer_encoder import TransformerEncoder
    return TransformerEncoder(encoder_id=enc_id, **encoder_cfg.params)
  
  else:
    raise ValueError(f"Encoder Type '{encoder_cfg.type}' not supported.")

def load_decoder(decoder_cfg):
    # # TODO: better way
    # if isinstance(encoder, nn.Sequential):
    #     n_out_features = encoder[-1].n_out_features
    # else:
    #     n_out_features = encoder.n_out_features

    # n_out_features = encoder.n_out_features

    if decoder_cfg.type == "TransformerDecoder":
      from .continuous.decoder.transformer_decoder import TransformerDecoder
      return TransformerDecoder(**decoder_cfg.params)

    else:
        raise ValueError(f"Decoder Type '{decoder_cfg.type}' not supported.")

def get_module_list(config, type="encoder"):
  module_list = nn.ModuleList()
  i=1
  while True:
    if hasattr(config.encoder_seq, type + str(i)):
      module_list.append(
        load_encoder(
          getattr(config.encoder_seq, type + str(i)),
          type + str(i)
        )
      )
      i+=1
    else:
      break
  return module_list

def get_cslr_model(config):
  encoder_seq = get_module_list(config)
  from .csl_network import CSLRModel
  return CSLRModel(encoder_seq, config, config.num_classes_gloss)

def get_slt_model(config):
  encoder_seq = get_module_list(config)
  decoder = load_decoder(config.decoder)
  
  return encoder_seq, decoder
  
  
