import os
import os
from enum import Enum
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader

from siren.utils import get_mgrid


class ImageEncoding(Enum):
    discrete: str = "discrete"
    continuous: str = "continuous"

class SIRENDataset(Dataset):
    def __init__(self, imgs_dir: Path, sidelength: int, image_encoding: ImageEncoding, transform=None):
        self.imgs_dir = imgs_dir
        self.imgs_paths = os.listdir(imgs_dir)
        self.class_coords = torch.linspace(-1,1,len(self.imgs_paths))
        self.transform = transform
        self.sidelength = sidelength
        self.image_encoding = image_encoding

    def __getitem__(self, index: int):
        img = pil_loader(self.imgs_dir / self.imgs_paths[index])
        if self.transform is not None:
            img = self.transform(img)

        pixels = img.permute(1, 2, 0).view(-1, 3)
        coords = generate_coords(index, self.sidelength, self.image_encoding, self.class_coords)
        return index, coords, pixels


    def __len__(self):
        return len(self.imgs_paths)


def generate_coords(img_index: int, sidelength: int, image_encoding: ImageEncoding, class_embeddings=None) -> torch.Tensor:
    coords = get_mgrid(sidelength, 2)

    if image_encoding == ImageEncoding("continuous"):
        img_index = class_embeddings[img_index]

    img_index = torch.tensor([img_index]).repeat(len(coords)).unsqueeze(1)

    coords = torch.cat((img_index, coords), dim=1)
    return coords




class OLSDataset(Dataset):
    """Dataset that is tailored to the OLS model"""
    def __init__(self, imgs_dir, sidelength: int, transform=None):
        self.imgs_dir = imgs_dir
        self.imgs_paths = os.listdir(imgs_dir)
        self.class_coords = torch.linspace(-1,1,len(self.imgs_paths))
        self.transform = transform
        self.sidelength = sidelength

    def __getitem__(self, index: int):
        img = pil_loader(self.imgs_dir / self.imgs_paths[index])
        if self.transform is not None:
            img = self.transform(img)

        pixels = img.permute(1, 2, 0).view(-1, 3)
        coords = get_mgrid(self.sidelength, 2)

        return index, coords, pixels



    def __len__(self):
        return len(self.imgs_paths)