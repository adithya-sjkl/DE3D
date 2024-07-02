import os
from glob import glob
import numpy as np
import torch
import monai
import nibabel
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    Resize,
    ToTensor,
    Spacing,
    Orientation,
    ScaleIntensityRange,
    CropForeground,
    RandFlip,
    RandRotate,

)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism
import random


path = "/Users/adithyasjith/Documents/Code/DE3D/Data"

class NiftiClassificationDataset(Dataset):
    def __init__(self, image_folder_disease, image_folder_healthy, transform, seed=None):
        self.image_folder_disease = image_folder_disease
        self.image_folder_healthy = image_folder_healthy
        self.transform = transform

        # List all NIfTI files
        self.disease_files = glob(os.path.join(image_folder_disease, "*.nii"))
        self.healthy_files = glob(os.path.join(image_folder_healthy, "*.nii"))
        self.total_files = self.disease_files + self.healthy_files
        self.labels = np.array([1] * len(self.disease_files) + [0] * len(self.healthy_files), dtype=np.int64)
        
        # Create a list of indices
        self.indices = list(range(len(self.total_files)))
        
        # Shuffle the indices
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.indices)
    def __len__(self):
        return len(self.total_files)

    def __getitem__(self, index):
        index = self.indices[index]
        image_file = self.total_files[index]
        label = self.labels[index]

        # Load NIfTI image
        image = self.transform(image_file)

        return image, label


def prepare(healthy_dir,disease_dir, pixdim=(1.5, 1.5, 1.0), a_min=-200, a_max=200, spatial_size=[224,224,224], batch_size=1):
    """
    This function is for preprocessing, it contains only the basic transforms, but you can add more operations that you
    find in the Monai documentation.
    https://monai.io/docs.html
    """

    set_determinism(seed=0)

    train_transforms = Compose(
        [
            LoadImage(),
            EnsureChannelFirst(),
            Spacing(pixdim=pixdim, mode=("bilinear")),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            CropForeground(),
            Resize(spatial_size=spatial_size),
            RandFlip(spatial_axis=0, prob=0.5),
            RandRotate(range_x=15, range_y=15, range_z=15, prob=0.5),
            ToTensor(),

        ]
    )

    dataset = NiftiClassificationDataset(image_folder_disease=disease_dir, image_folder_healthy=healthy_dir, transform=train_transforms, seed=42)

    train_dataset,val_dataset,test_dataset = monai.data.utils.partition_dataset(dataset, ratios=[0.8, 0.1, 0.1])

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
