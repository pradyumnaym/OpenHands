import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from .base_encoder import EncoderModule

'''
2 types of encoders:
IMAGE ENCODERS: Take ip of shape (batch_size, max_seqlen, n_channels, height, width)
FEATURE ENCODERS: Take ip of shape (batch_size, max_seqlen, n_feature)

Each batch item is a sequence. Sequences will be padded to max_seq_len of the batch.
So every encoder should return: 
  encoded features
  updated seq_lens
  loss_terms
'''
## keeping it here for now to identify the source from where the class was copied
## Since ResNet18 is a general module, it should ideally be in some common file
## instead of this specific TLP file. Or maybe just import from torch.utils directly

## to implement parts common to all Encoder Modules

class TLPResnet18(EncoderModule):
  def __init__(self,**kwargs):
    super(TLPResnet18, self).__init__(**kwargs)
    self.conv2d = resnet18(pretrained=False) ## make this true when training
    self.conv2d.fc = Identity()
    self.out_size = 512
    
  def masked_bn(self, inputs, len_x):
    def pad(tensor, length):
      return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

    x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
    x = self.conv2d(x)
    x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                    for idx, lgt in enumerate(len_x)])
    return x

  def forward(self, x, len_x):
    '''
    INPUT:
    x : [batch_size, max_seq_len (in batch), n_channels, height, width]
    len_x : [batch_size]; seq_len of each batch member
    OUTPUT:
    framewise: [batch_size, max_seq_len, hidden_size (=512)] 
    '''
    batch, temp, channel, height, width = x.shape
    inputs = x.reshape(batch * temp, channel, height, width)
    framewise = self.masked_bn(inputs, len_x) # [batch_size*max_seq_len, hidden_size (=512)]
    framewise = framewise.reshape(batch, temp, -1) #.transpose(1, 2) #do this inside TempConv

    return framewise, len_x, self.internal_losses

class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x



class Temporal_LiftPool(nn.Module):
  def __init__(self, input_size, kernel_size=2):
    super(Temporal_LiftPool, self).__init__()
    self.kernel_size = kernel_size
    self.predictor = nn.Sequential(
      nn.Conv1d(input_size, input_size, kernel_size=3, stride=1, padding=1, groups=input_size), 
      nn.ReLU(inplace=True),   
      nn.Conv1d(input_size, input_size, kernel_size=1, stride=1, padding=0),
      nn.Tanh(),    
                              )

    self.updater = nn.Sequential(
      nn.Conv1d(input_size, input_size, kernel_size=3, stride=1, padding=1, groups=input_size),
      nn.ReLU(inplace=True),   
      nn.Conv1d(input_size, input_size, kernel_size=1, stride=1, padding=0),
      nn.Tanh(),    
                              )
    self.predictor[2].weight.data.fill_(0.0)
    self.updater[2].weight.data.fill_(0.0)
    self.weight1 = Local_Weighting(input_size)
    self.weight2 = Local_Weighting(input_size)

  def forward(self, x):
    B, C, T= x.size()
    Xe = x[:,:,:T:self.kernel_size]
    Xo = x[:,:,1:T:self.kernel_size]
    d = Xo - self.predictor(Xe)
    s = Xe + self.updater(d)
    loss_u = torch.norm(s-Xo, p=2)
    loss_p = torch.norm(d, p=2)
    s = torch.cat((x[:,:,:0:self.kernel_size], s, x[:,:,T::self.kernel_size]),2)
    return self.weight1(s)+self.weight2(d), loss_u, loss_p

class Local_Weighting(nn.Module):
  def __init__(self, input_size ):
    super(Local_Weighting, self).__init__()
    self.conv = nn.Conv1d(input_size, input_size, kernel_size=5, stride=1, padding=2)
    self.insnorm = nn.InstanceNorm1d(input_size, affine=True)
    self.conv.weight.data.fill_(0.0)

  def forward(self, x):
    out = self.conv(x)
    return x + x*(F.sigmoid(self.insnorm(out))-0.5)

class TemporalConv(EncoderModule):
  def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1, **kwargs):
    super(TemporalConv, self).__init__(**kwargs)
    self.use_bn = use_bn
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.out_size = hidden_size
    self.num_classes = num_classes
    self.conv_type = conv_type
    

    if self.conv_type == 0:
      self.kernel_size = ['K3']
    elif self.conv_type == 1:
      self.kernel_size = ['K5', "P2"]
      self.strides = [0]
    elif self.conv_type == 2:
      self.kernel_size = ['K5', "P2", 'K5', "P2"]
      self.strides = [4,0]


    self.temporal_conv = nn.ModuleList([])
    #nums = 0
    for layer_idx, ks in enumerate(self.kernel_size):
      input_sz = self.input_size if layer_idx == 0 else self.hidden_size
      if ks[0] == 'P':
        self.temporal_conv.append(Temporal_LiftPool(input_size=input_sz, kernel_size=int(ks[1])))
        
      elif ks[0] == 'K':
        self.temporal_conv.append(
          nn.Sequential(
          nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0),
          nn.BatchNorm1d(self.hidden_size),
          nn.ReLU(inplace=True),
          )
        )

  def update_lgt(self, feat_len):
    for ks in self.kernel_size:
      if ks[0] == 'P':
        feat_len //= int(ks[1])
      else:
        feat_len -= int(ks[1]) - 1
    return feat_len

  def forward(self, frame_feat, lgt):
    visual_feat = frame_feat.transpose(1,2) # visual_feat needs (B,H,MSL)
    loss_LiftPool_u = 0
    loss_LiftPool_p = 0
    i = 0
    for tempconv in self.temporal_conv:
      if isinstance(tempconv, Temporal_LiftPool):
        visual_feat, loss_u, loss_d = tempconv(visual_feat) #self.strides[i])
        i +=1
        loss_LiftPool_u += loss_u
        loss_LiftPool_p += loss_d
      else:
        visual_feat = tempconv(visual_feat)
    lgt = self.update_lgt(lgt)
    
    ## LiftPool losses
    self.internal_losses.update({
      "Cu": loss_LiftPool_u,
      "Cp": loss_LiftPool_p,
    })
    
    return visual_feat.transpose(1,2), lgt, self.internal_losses
    


class BiLSTMLayer(EncoderModule):
  def __init__(self, input_size, debug=False, hidden_size=1024, num_layers=1, dropout=0.3,
              bidirectional=True, rnn_type='LSTM', **kwargs):
    super(BiLSTMLayer, self).__init__(**kwargs)

    self.dropout = dropout
    self.num_layers = num_layers
    self.input_size = input_size
    self.bidirectional = bidirectional
    self.num_directions = 2 if bidirectional else 1
    self.hidden_size = int(hidden_size / self.num_directions)
    self.out_size = int(2*self.hidden_size)
    self.rnn_type = rnn_type
    self.debug = debug
    self.rnn = getattr(nn, self.rnn_type)(
        input_size=self.input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        dropout=self.dropout,
        bidirectional=self.bidirectional)
    # for name, param in self.rnn.named_parameters():
    #     if name[:6] == 'weight':
    #         nn.init.orthogonal_(param)

  def forward(self, src_feats, src_lens, hidden=None):
    """
    Args:
      - src_feats: (B, msl, D)
      - src_lens: (B)
    Returns:
      - outputs: (max_src_len, batch_size, hidden_size * num_directions)
      - hidden : (num_layers, batch_size, hidden_size * num_directions)
    """
    # (max_src_len, batch_size, D)
    packed_emb = nn.utils.rnn.pack_padded_sequence(src_feats.transpose(0,1), src_lens.cpu())

    # rnn(gru) returns:
    # - packed_outputs: shape same as packed_emb
    # - hidden: (num_layers * num_directions, batch_size, hidden_size)
    if hidden is not None and self.rnn_type == 'LSTM':
      half = int(hidden.size(0) / 2)
      hidden = (hidden[:half], hidden[half:])
    packed_outputs, hidden = self.rnn(packed_emb, hidden)

    # outputs: (max_src_len, batch_size, hidden_size * num_directions)
    rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

    if self.bidirectional:
      # (num_layers * num_directions, batch_size, hidden_size)
      # => (num_layers, batch_size, hidden_size * num_directions)
      hidden = self._cat_directions(hidden)

    if isinstance(hidden, tuple):
      # cat hidden and cell states
      hidden = torch.cat(hidden, 0)

    # original return values
    # return {
    #   "predictions": rnn_outputs,
    #   "hidden": hidden
    # }
    return rnn_outputs.transpose(0,1), src_lens, self.internal_losses
    


  def _cat_directions(self, hidden):
    """ If the encoder is bidirectional, do the following transformation.
      Ref: https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/DecoderRNN.py#L176
      -----------------------------------------------------------
      In: (num_layers * num_directions, batch_size, hidden_size)
      (ex: num_layers=2, num_directions=2)

      layer 1: forward__hidden(1)
      layer 1: backward_hidden(1)
      layer 2: forward__hidden(2)
      layer 2: backward_hidden(2)

      -----------------------------------------------------------
      Out: (num_layers, batch_size, hidden_size * num_directions)

      layer 1: forward__hidden(1) backward_hidden(1)
      layer 2: forward__hidden(2) backward_hidden(2)
    """

    def _cat(h):
      return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

    if isinstance(hidden, tuple):
      # LSTM hidden contains a tuple (hidden state, cell state)
      hidden = tuple([_cat(h) for h in hidden])
    else:
      # GRU hidden
      hidden = _cat(hidden)

    return hidden


