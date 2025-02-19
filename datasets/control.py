from typing import List, Tuple, Union, Literal
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from Register import Registers
from datasets.base import ImagePathDataset, ImageNPZDataset
from datasets.utils import get_image_paths_from_dir
from datasets.rcdm import LFW, FFHQ
from PIL import Image
import cv2
import os

class RCDMControlDataset(Dataset):
  '''
  dataset for RCDM.
  Args:
    handle: tuple of dataset type and path to the npz file.
    facebase: path to similar images of `handle`.
  '''
  def __init__(self, handle: Tuple[Literal["LFW", "FFHQ"], str], facebase: str, img_size=128):
    self.transform = transforms.Compose([
      transforms.Resize((img_size, img_size)),
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
    ])
    hd_path = handle[1]
    self.dataset_type = handle[0]
    if handle[0] == "FFHQ":
      ds = FFHQ(hd_path, self.transform)
    elif handle[0] == "LFW":
      ds = LFW(hd_path, self.transform)
    else:
      raise NotImplementedError("only support LFW and FFHQ target")
    self.target = ds
    self.base_images = get_image_paths_from_dir(facebase)
    assert len(self.target) == len(self.base_images), "source and target must have the same amount"
  
  def __getitem__(self, index):
    target_img, feature, label = self.target.__getitem__(index)
    base = Image.open(self.base_images[index])
    base = self.transform(base)
    return (
      (target_img, f"{self.dataset_type}_{index:06}"), # x
      (base, f"{self.dataset_type}_similar_{index:06}"), # x_cond
      feature) # control
  
  def __len__(self):
    return self.target.__len__()