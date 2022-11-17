import os
import re
import torchtext
import pandas as pd
from .base import BaseContinuousDataset
from ..data_readers import load_frames_from_video

class DGSDataset(BaseContinuousDataset):
    """
    The German DGS Corpus from the paper:
    
    `Extending the Public DGS Corpus in Size and Depth <https://aclanthology.org/2020.signlang-1.12/>`_
    """

    lang_code = "gsg"

    def __init__(
        self,
        use_english = False,
        train_file = None,
        **kwargs
    ):
        self.use_english = use_english
        self.train_file = train_file

        self.gloss_tokenizer = torchtext.data.utils.get_tokenizer(None)
        self.text_tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='de_core_news_sm')

        self.gloss_key = "GLOSS_SIGN" if self.use_english else "GLOSS_SIGN_GERMAN"
        self.text_key = "TEXT" if self.use_english else "TEXT_GERMAN"

        super().__init__(**kwargs)

    def read_glosses(self):
        glosses = list()
        texts = list()
        df = pd.read_csv(self.train_file, escapechar="\\", delimiter='|').fillna('')
        
        for i in range(len(df)):
            glosses.append(self.gloss_tokenizer(df[self.gloss_key][i]))
            texts.append(self.text_tokenizer(df[self.text_key][i]))
        
        self.build_vocab_from_iterators(glosses, texts)

    def read_original_dataset(self):
        df = pd.read_csv(self.split_file, escapechar="\\", delimiter='|').fillna('')

        for i in range(len(df)):
            instance_entry = df["VIDEO"][i], self.gloss_tokenizer(df[self.gloss_key][i]), self.text_tokenizer(df[self.text_key][i])
            self.data.append(instance_entry)

    def read_video_data(self, index):
        video_name, gloss_seq, text_seq = self.data[index]
        video_path = os.path.join(self.root_dir, video_name)
        imgs = load_frames_from_video(video_path)
        return imgs, gloss_seq, text_seq, video_name
