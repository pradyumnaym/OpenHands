import os
import pandas as pd
import torchtext
from .base import BaseContinuousDataset
from ..data_readers import load_frames_from_folder

class Phoenix14Dataset(BaseContinuousDataset):
  """
    German Continuous Sign language dataset from the paper:
    O. Koller, J. Forster, and H. Ney. Continuous sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers. 
    https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/
  """

  # lang_code = "ins" <= what is the purpose of this?
  def __init__(self, train_file = None,**kwargs):
        self.train_file = train_file
        self.gloss_tokenizer = torchtext.data.utils.get_tokenizer(None)
        super().__init__(**kwargs)


  def read_glosses(self):
    glosses = list() ## should only contain train_glosses or all glosses
    df = pd.read_csv(self.train_file, delimiter='|').fillna('')
    
    for i in range(len(df)):
      glosses.append(self.gloss_tokenizer(df['annotation'][i]))
    
    self.build_vocab_from_iterators(glosses, list()) # no translation data in phoenix14
    
    

  def read_original_dataset(self):
    df = pd.read_csv(self.split_file, delimiter='|').fillna('')
    for i in range(len(df)):
      vid_dir = os.path.join('features/fullFrame-256x256px/', 
                            'dev' if self.splits == 'val' else self.splits, 
                            df["folder"][i].rsplit('/',1)[0])
      
      # TODO: Replace extension with pkl for pose modality?
      if "rgb" in self.modality and not os.path.isdir(os.path.join(self.root_dir, vid_dir)):
        print(f"Video not found: {os.path.join(self.root_dir, vid_dir)}")
        continue
      instance_entry = vid_dir, self.gloss_tokenizer(df['annotation'][i]), list()
      
      
      self.data.append(instance_entry)

  def read_video_data(self, index):
    video_dir, gloss_seq, text_seq = self.data[index]
    imgs = load_frames_from_folder(os.path.join(self.root_dir, video_dir), pattern='*.png')
    return imgs, gloss_seq, text_seq, video_dir

