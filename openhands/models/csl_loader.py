import omegaconf
import torch.nn as nn
import hydra

def load_encoder(encoder_cfg, enc_id):
  if encoder_cfg.type == "custom":
    return hydra.utils.instantiate(**encoder_cfg.params)
  
  if encoder_cfg.type == "ResNet18":
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
  else:
    raise ValueError(f"Encoder Type '{encoder_cfg.type}' not supported.")

def load_decoder(decoder_cfg, num_class, encoder):
    # # TODO: better way
    # if isinstance(encoder, nn.Sequential):
    #     n_out_features = encoder[-1].n_out_features
    # else:
    #     n_out_features = encoder.n_out_features

    n_out_features = encoder.n_out_features

    if decoder_cfg.type == "fc":
        from .decoder.fc import FC

        return FC(
            n_features=n_out_features, num_class=num_class, **decoder_cfg.params
        )
    elif decoder_cfg.type == "rnn":
        from .decoder.rnn import RNNClassifier

        return RNNClassifier(
            n_features=n_out_features, num_class=num_class, **decoder_cfg.params
        )
    elif decoder_cfg.type == "bert":
        from .decoder.bert_hf import BERT

        return BERT(
            n_features=n_out_features,
            num_class=num_class,
            config=decoder_cfg.params,
        )
    elif decoder_cfg.type == "fine_tuner":
        from .decoder.fine_tuner import FineTuner

        return FineTuner(
            n_features=n_out_features, num_class=num_class, **decoder_cfg.params
        )
    else:
        raise ValueError(f"Decoder Type '{decoder_cfg.type}' not supported.")

def load_ssl_backbone(cfg, in_channels, num_class):
    if cfg.type == 'dpc':
        from .ssl.dpc_rnn import DPC_RNN_Finetuner, load_weights_from_pretrained
        # Load pretraining config
        pretraining_cfg = omegaconf.OmegaConf.load(cfg.load_from.cfg_file)
        pretraining_cfg.in_channels = in_channels
        # Create model
        model = DPC_RNN_Finetuner(num_class=num_class, **pretraining_cfg.model)
        # Load weights
        model = load_weights_from_pretrained(model, cfg.load_from.ckpt)
        return model
    else:
        raise ValueError(f"SSL Type '{cfg.type}' not supported.")

def get_cslr_model(config):
  # if "pretrained" in config:
  #     # Load self-supervised backbone
  #     return load_ssl_backbone(config.pretrained, in_channels, num_class)
  encoder_seq = nn.ModuleList()
  i=1
  while True:
    if hasattr(config.encoder_seq, f'encoder{i}'):
      encoder_seq.append(load_encoder(getattr(config.encoder_seq, f'encoder{i}'), f'encoder{i}'))
      i+=1
    else:
      break
  
  from .csl_network import CSLRModel
  return CSLRModel(encoder_seq, config, config.num_classes_gloss)
  
