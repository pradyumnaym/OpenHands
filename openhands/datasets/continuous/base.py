import torch
import torch.nn.functional as F
import torchvision
import torchtext
import pickle
import albumentations as A
import numpy as np
import pandas as pd
import os, warnings
from .vocabulary import (
    get_vocabulary_from_iter,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN
)
from ..video_transforms import *
from ..data_readers import *

class BaseContinuousDataset(torch.utils.data.Dataset):
    """
    This module provides the datasets for Continuous Sign Language Recognition/Translation.
    Do not instantiate this class
    """

    lang_code = None
    # Get language from here:
    # https://iso639-3.sil.org/code_tables/639/data?title=&field_iso639_cd_st_mmbrshp_639_1_tid=94671&name_3=sign+language&field_iso639_element_scope_tid=All&field_iso639_language_type_tid=All&items_per_page=200

    ASSETS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

    def __init__(
        self,
        root_dir,
        split_file=None,
        class_mappings_file_path=None,
        normalized_class_mappings_file=None,
        splits=["train"],
        modality="rgb",
        transforms="default",
        cv_resize_dims=(264, 264),
        pose_use_confidence_scores=False,
        pose_use_z_axis=False,
        inference_mode=False,
        only_metadata=False, # Does not load data files if `True`
        multilingual=False,
        languages=None,
        language_set=None,
        
        # Windowing
        seq_len=1, # No. of frames per window
        num_seq=1, # No. of windows
    ):
        super().__init__()

        self.split_file = split_file
        self.root_dir = root_dir
        self.class_mappings_file_path = class_mappings_file_path
        self.splits = splits
        self.modality = modality
        self.multilingual = multilingual
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.languages=languages
        self.language_set=language_set
        
        self.gloss_vocab = None
        self.text_vocab = None
        self.read_glosses()
        print(f"Found {len(self.gloss_vocab)} tokens in the gloss vocabulary.")
        print(f"Found {len(self.text_vocab)} tokens in the text vocabulary.")

        self.inference_mode = inference_mode
        self.only_metadata = only_metadata

        if not only_metadata:
            self.data = []
            
            if inference_mode:
                # Will have null labels
                self.enumerate_data_files(self.root_dir)
            else:
                self.read_original_dataset()
            if not self.data:
                raise RuntimeError("No data found")

        self.cv_resize_dims = cv_resize_dims
        self.pose_use_confidence_scores = pose_use_confidence_scores
        self.pose_use_z_axis = pose_use_z_axis

        if "rgb" in modality:
            self.in_channels = 3
            if modality == "rgbd":
                self.in_channels += 1

            self.__getitem = self.__getitem_video

        elif modality == "pose":
            self.in_channels = 4
            if not self.pose_use_confidence_scores:
                self.in_channels -= 1
            if not self.pose_use_z_axis:
                self.in_channels -= 1

            self.__getitem = self.__getitem_pose

        else:
            exit(f"ERROR: Modality `{modality}` not supported") 

        self.setup_transforms(modality, transforms)

    def setup_transforms(self, modality, transforms):
        if "rgb" in modality:
            if transforms == "default":
                albumentation_transforms = A.Compose(
                    [
                        A.ShiftScaleRotate(
                            shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                        ),
                        A.ChannelDropout(p=0.1),
                        A.RandomRain(p=0.1),
                        A.GridDistortion(p=0.3),
                    ]
                )
                self.transforms = torchvision.transforms.Compose(
                    [
                        Albumentations2DTo3D(albumentation_transforms),
                        NumpyToTensor(),
                        RandomTemporalSubsample(16),
                        torchvision.transforms.Resize(
                            (self.cv_resize_dims[0], self.cv_resize_dims[1])
                        ),
                        torchvision.transforms.RandomCrop(
                            (self.cv_resize_dims[0], self.cv_resize_dims[1])
                        ),
                        torchvision.transforms.RandomHorizontalFlip(p=0.5),
                        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        TCHW2CTHW(),
                    ]
                )
            elif transforms:
                self.transforms = transforms
            else:
                self.transforms = torchvision.transforms.Compose(
                    [
                        NumpyToTensor(),
                        # THWC2CTHW(),
                        THWC2TCHW(),
                        torchvision.transforms.Resize(
                            (self.cv_resize_dims[0], self.cv_resize_dims[1])
                        ),
                        TCHW2CTHW(),
                    ]
                )
        elif "pose" in modality:
            if transforms == "default":
                transforms = None
            self.transforms = transforms

        self.transforms_gloss = torchtext.transforms.Sequential(
            torchtext.transforms.AddToken(BOS_TOKEN),
            torchtext.transforms.AddToken(EOS_TOKEN, begin=False),
            torchtext.transforms.VocabTransform(self.gloss_vocab),
        )

        self.transforms_text = torchtext.transforms.Sequential(
            torchtext.transforms.AddToken(BOS_TOKEN),
            torchtext.transforms.AddToken(EOS_TOKEN, begin=False),
            torchtext.transforms.VocabTransform(self.text_vocab),
        )

    @property
    def vocab_size_gloss(self):
        return len(self.gloss_vocab)

    @property
    def vocab_size_text(self):
        return len(self.text_vocab)

    def read_glosses(self):
        """
        Implement this method to construct `self.glosses[]`
        """
        raise NotImplementedError

    def build_vocab_from_iterators(self, glosses, texts):
        """
        Builds a torchtext.vocab.Vocab object for glosses and text, based on the tokens
        generated by the iterators. To be called by the subclass with apropriate token-
        -ization scheme.

        Sets the self.gloss_vocab and self.text_vocab attributes.
        """
        self.gloss_vocab = get_vocabulary_from_iter(iter(glosses))
        self.text_vocab  = get_vocabulary_from_iter(iter(texts))

    def read_original_dataset(self):
        """
        Implement this method to read (video_name/video_folder, classification_label)
        into self.data[]
        """
        raise NotImplementedError
    
    def enumerate_data_files(self, dir):
        """
        Lists the video files from given directory.
        - If pose modality, generate `.pkl` files for all videos in folder.
          - If no videos present, check if some `.pkl` files already exist
        """
        files = list_all_videos(dir)

        if self.modality == "pose":
            holistic = None
            pose_files = []

            for video_file in files:
                pose_file = os.path.splitext(video_file)[0] + ".pkl"
                if not os.path.isfile(pose_file):
                    # If pose is not cached, generate and store it.
                    if not holistic:
                        # Create MediaPipe instance
                        from ..pipelines.generate_pose import MediaPipePoseGenerator
                        holistic = MediaPipePoseGenerator()
                    # Dump keypoints
                    frames = load_frames_from_video(video_file)
                    holistic.generate_keypoints_for_frames(frames, pose_file)
                    
                pose_files.append(pose_file)
            
            if not pose_files:
                pose_files = list_all_files(dir, extensions=[".pkl"])
            
            files = pose_files
        
        if not files:
            raise RuntimeError(f"No files found in {dir}")
        
        self.data = [(f, -1) for f in files]
        # -1 means invalid label_id

    def __len__(self):
        return len(self.data)

    def load_pose_from_path(self, path):
        """
        Load dumped pose keypoints.
        Should contain: {
            "keypoints" of shape (T, V, C),
            "confidences" of shape (T, V)
        }
        """
        pose_data = pickle.load(open(path, "rb"))
        return pose_data

    def read_video_data(self, index):
        """
        Extend this method for dataset-specific formats
        """
        video_path = self.data[index][0]
        label = self.data[index][1]
        imgs = load_frames_from_video(video_path)
        return imgs, label, video_name

    def __getitem_video(self, index):
        if self.inference_mode:
            imgs, gloss_seq, text_seq, video_name = super().read_video_data(index)
        else:
            imgs, gloss_seq, text_seq, video_name = self.read_video_data(index)
        # imgs shape: (T, H, W, C)

        if self.transforms is not None:
            imgs = self.transforms(imgs)
        
        gloss_seq = self.transforms_gloss(gloss_seq)
        text_seq = self.transforms_text(text_seq)
        print(imgs.size())
        return {
            "frames": imgs,
            "gloss": gloss_seq,
            "text": text_seq,
            "file": video_name,
            "dataset_name": None
        }

    #@staticmethod
    def collate_fn(self, batch_list):
        if "num_windows" in batch_list[0]:
            # Padding not required for windowed models
            frames=[x["frames"] for x in batch_list]
        else:
            max_frames = max([x["frames"].shape[1] for x in batch_list])
            # Pad the temporal dimension to `max_frames` for all videos
            # Assumes each instance of shape: (C, T, V) 
            # TODO: Handle videos (C,T,H,W)
            if self.modality == "pose":
                frames = [
                    F.pad(x["frames"], (0, 0, 0, max_frames - x["frames"].shape[1], 0, 0))
                    for i, x in enumerate(batch_list)
                ]
            elif self.modality == "rgb":
                frames = [
                    F.pad(x["frames"], (0, 0, 0, 0, 0, max_frames - x["frames"].shape[1]))
                    for i, x in enumerate(batch_list)
                ]
            else:
                raise NotImplementedError("Modality not implemented!")

        frames = torch.stack(frames, dim=0)

        return dict(
            frames=frames,
            gloss=torchtext.functional.to_tensor([x["gloss"] for x in batch_list], padding_value=self.gloss_vocab[PAD_TOKEN]),
            text=torchtext.functional.to_tensor([x["text"] for x in batch_list], padding_value=self.text_vocab[PAD_TOKEN]),
            files=[x["file"] for x in batch_list],
            dataset_names=[x["dataset_name"] for x in batch_list]
        )

    def read_pose_data(self, index):
        _, gloss_seq, text_seq = self.data[index]
        if self.inference_mode:
            pose_path = self.data[index][0]
        else:
            video_name = self.data[index][0]
            
            video_path = os.path.join(self.root_dir, video_name)
            # print("--------------279",self.root_dir)
            # print("---------280",video_name)
            # If `video_path` is folder of frames from which pose was dumped, keep it as it is.
            # Otherwise, just remove the video extension
            pose_path = (
                video_path if os.path.isdir(video_path) else os.path.splitext(video_path)[0]
            )
            pose_path = pose_path + ".pkl"
            
        pose_data = self.load_pose_from_path(pose_path)

        pose_data["gloss_seq"] = gloss_seq
        pose_data["text_seq"] = text_seq

        if self.multilingual:
            # if `ConcatDataset` is used, it has extra entries for following:
            pose_data["lang_code"] = self.data[index][2]
            pose_data["dataset_name"] = self.data[index][3]
        return pose_data, pose_path

    def __getitem_pose(self, index):
        """
        Returns
        C - num channels
        T - num frames
        V - num vertices
        """
        data, path = self.read_pose_data(index)
        # imgs shape: (T, V, C)
        kps = data["keypoints"]
        scores = data["confidences"]

        if not self.pose_use_z_axis:
            kps = kps[:, :, :2]

        if self.pose_use_confidence_scores:
            kps = np.concatenate([kps, np.expand_dims(scores, axis=-1)], axis=-1)

        kps = np.asarray(kps, dtype=np.float32)
        data = {
            "frames": torch.tensor(kps).permute(2, 0, 1),  # (C, T, V)
            "gloss": self.transforms_gloss(data["gloss_seq"]),
            "text": self.transforms_text(data["text_seq"]),
            "file": path,
            "lang_code": data["lang_code"] if self.multilingual else None, # Required for lang_token prepend
            "dataset_name": data["dataset_name"] if self.multilingual else None, # Required to calc dataset-wise accuracy
        }

        if self.transforms is not None:
            data = self.transforms(data)
        
        if self.seq_len > 1 and self.num_seq > 1:
            data["num_windows"] = self.num_seq
            kps = data["frames"].permute(1, 2, 0).numpy() # CTV->TVC
            if kps.shape[0] < self.seq_len * self.num_seq:
                pad_kps = np.zeros(
                    ((self.seq_len * self.num_seq) - kps.shape[0], *kps.shape[1:])
                )
                kps = np.concatenate([pad_kps, kps])

            elif kps.shape[0] > self.seq_len * self.num_seq:
                kps = kps[: self.seq_len * self.num_seq, ...]

            SL = kps.shape[0]
            clips = []
            i = 0
            while i + self.seq_len <= SL:
                clips.append(torch.tensor(kps[i : i + self.seq_len, ...], dtype=torch.float32))
                i += self.seq_len

            t_seq = torch.stack(clips, 0)
            data["frames"] = t_seq.permute(0, 3, 1, 2) # WTVC->WCTV

        return data

    def __getitem__(self, index):
        return self.__getitem(index)
