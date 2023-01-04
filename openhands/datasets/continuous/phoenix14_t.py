import os
import re
import pandas as pd
import torchtext

from itertools import groupby

from .base import BaseContinuousDataset
from ..data_readers import load_frames_from_folder

class Phoenix14TDataset(BaseContinuousDataset):
  """
    German Continuous Sign language dataset from the paper:
    Necati Cihan Camgöz, Simon Hadfield, Oscar Koller, Hermann Ney, Richard Bowden, Neural Sign Language Translation
    https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/
  """

  # lang_code = "ins" <= what is the purpose of this?
  def __init__(self, train_file = None,**kwargs):
    self.train_file = train_file
    self.gloss_tokenizer = torchtext.data.utils.get_tokenizer(None)
    self.text_tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='de_core_news_sm')
    super().__init__(**kwargs)

  def read_glosses(self):
    glosses, texts = list(), list()
    df = pd.read_csv(self.train_file, delimiter='|').fillna('')
    for i in range(len(df)):
      glosses.append(self.gloss_tokenizer(df['orth'][i]))
      texts.append(self.text_tokenizer(df['translation'][i]))
    
    self.build_vocab_from_iterators(glosses, texts)
    
    

  def read_original_dataset(self):
    df = pd.read_csv(self.split_file, delimiter='|').fillna('')
    for i in range(len(df)):
      vid_dir = os.path.join('features/fullFrame-210x260px/', 
                            'dev' if self.splits == 'val' else self.splits, 
                            df['video'][i].rsplit('/',2)[0])
      
      ## should pass gloss_id as second part of instance_entry, but glosses not available
      instance_entry = vid_dir, self.gloss_tokenizer(df['orth'][i]), self.text_tokenizer(df['translation'][i])
      # TODO: Replace extension with pkl for pose modality?
      if "rgb" in self.modality and not os.path.isdir(os.path.join(self.root_dir, vid_dir)):
        print(f"Video Folder not found: {os.path.join(self.root_dir, vid_dir)}")
        continue
      
      self.data.append(instance_entry)

  def read_video_data(self, index):
    video_dir, gloss_seq, text_seq = self.data[index]
    imgs = load_frames_from_folder(os.path.join(self.root_dir, video_dir), pattern='*.png')
    return imgs, gloss_seq, text_seq, video_dir

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

