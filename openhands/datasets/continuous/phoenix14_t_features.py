import os
import re
import gzip
import torch
import pickle
import pandas as pd
import torchtext

from itertools import groupby

from .base import BaseContinuousDataset
from ..data_readers import load_frames_from_folder

class Phoenix14TFeaturesDataset(BaseContinuousDataset):
  """
    German Continuous Sign language dataset from the paper:
    Necati Cihan Camgöz, Simon Hadfield, Oscar Koller, Hermann Ney, Richard Bowden, Neural Sign Language Translation
    https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/
  """

  # lang_code = "ins" <= what is the purpose of this?
  def __init__(self, train_file = None,**kwargs):
    self.train_file = train_file
    self.gloss_tokenizer = torchtext.data.utils.get_tokenizer(None)
    self.text_tokenizer = torchtext.data.utils.get_tokenizer(None)
    super().__init__(**kwargs)

  def read_glosses(self):
    glosses, texts = list(), list()
    data = self.load_dataset_file(self.train_file)
    for i in data:
      #signer independent split
      if i['signer'] == 'Signer05':
        continue
      glosses.append(self.gloss_tokenizer(i["gloss"]))
      texts.append(self.text_tokenizer(i["text"]))
    
    self.build_vocab_from_iterators(glosses, texts)

  def read_original_dataset(self):
    data = self.load_dataset_file(self.split_file)
    for i in data:
      if self.splits == 'train' and i['signer'] == 'Signer05':
        continue
      elif self.splits != 'train' and i['signer'] != 'Signer05':
        continue
      self.data.append((
        i["sign"], 
        self.gloss_tokenizer(i["gloss"]), 
        self.text_tokenizer(i["text"].strip())
      ))

  def read_video_data(self, index):
    imgs, gloss_seq, text_seq = self.data[index] #video is a (T, 1024) tensor - split it into individual frames
    return imgs, gloss_seq, text_seq, ""

  def clean_glosses(self, prediction):
    prediction = prediction.strip()
    prediction = re.sub(r"__LEFTHAND__", "", prediction)
    prediction = re.sub(r"__EPENTHESIS__", "", prediction)
    prediction = re.sub(r"__EMOTION__", "", prediction)
    prediction = re.sub(r"\b__[^_ ]*__\b", "", prediction)
    prediction = re.sub(r"\bloc-([^ ]*)\b", r"\1", prediction)
    prediction = re.sub(r"\bcl-([^ ]*)\b", r"\1", prediction)
    prediction = re.sub(r"\b([^ ]*)-PLUSPLUS\b", r"\1", prediction)
    prediction = re.sub(r"\b([A-Z][A-Z]*)RAUM\b", r"\1", prediction)
    prediction = re.sub(r"WIE AUSSEHEN", "WIE-AUSSEHEN", prediction)
    prediction = re.sub(r"^([A-Z]) ([A-Z][+ ])", r"\1+\2", prediction)
    prediction = re.sub(r"[ +]([A-Z]) ([A-Z]) ", r" \1+\2 ", prediction)
    prediction = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +]SCH) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +]NN) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +][A-Z]) (NN[ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +][A-Z]) ([A-Z])$", r"\1+\2", prediction)
    prediction = re.sub(r" +", " ", prediction)
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction)
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction)
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction)
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction)
    prediction = re.sub(r" +", " ", prediction)

    # Remove white spaces and repetitions
    prediction = " ".join(
        " ".join(i[0] for i in groupby(prediction.split(" "))).split()
    )
    prediction = prediction.strip()

    return prediction

  def load_dataset_file(self, filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

