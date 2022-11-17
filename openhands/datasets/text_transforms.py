import torch
import random
import math
import numpy as np
import random

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, text):
        for transform in self.transforms:
            text = transform(text)
        return text

