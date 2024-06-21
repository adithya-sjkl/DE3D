import numpy as np
import pandas as pd
import torch
import torchvision
import monai
from glob import glob
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,

)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism
import os

path = "/Users/adithyasjith/Documents/Code/DE3D/Data"

#testing