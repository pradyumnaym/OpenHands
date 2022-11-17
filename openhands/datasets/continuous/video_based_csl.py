import os
import re
import pandas as pd
from collections import defaultdict
from .base import BaseContinuousDataset
from ..data_readers import load_frames_from_video

class VideoBasedCSLDataset(BaseContinuousDataset):
    """
    The Video-Based CSL dataset from the paper:
    
    `Video-Based Sign Language Recognition without Temporal Segmentation <https://dl.acm.org/doi/pdf/10.5555/3504035.3504310>`_
    """

    lang_code = "csl"

    def read_glosses(self):

        pass
        # # Remove serial numbers from gloss names
        # # We are removing it after sorting, because the models we released have classes in the above order
        # self.glosses = [re.sub("\d+\.", '', gloss).strip().upper() for gloss in self.glosses]
        # # Nevermind, this creates issue at `read_original_dataset()`

    def read_original_dataset(self):
        with open(os.path.join(self.root_dir, "sentence_label.txt"), "r") as f:
            gloss_labels = f.readlines()
        
        for idx, sentence_id in enumerate(sorted(os.listdir("color"))):
            for video in sorted(os.listdir(os.path.join("color", sentence_id))):
                instance = os.path.join("color", sentence_id, video), gloss_labels[idx].rstrip('\n'), ""
                self.data.append(instance)

    def read_video_data(self, index):
        video_name, gloss_seq, text_seq = self.data[index]
        video_path = os.path.join(self.root_dir, video_name)
        imgs = load_frames_from_video(video_path)
        return imgs, gloss_seq, text_seq, video_name
