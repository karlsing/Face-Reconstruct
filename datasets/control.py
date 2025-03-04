import os
import cv2
from typing import List, Tuple, Union, Literal
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from Register import Registers
from datasets.base import ImagePathDataset, ImageNPZDataset
from datasets.utils import get_image_paths_from_dir
from datasets.rcdm import LFW, FFHQ
from PIL import Image

@Registers.datasets.register_with_name('custom_rcdm')
class RCDMControlDataset(Dataset):
  '''
  dataset for RCDM.
  Args:
    handle: tuple of dataset type and path to the npz file.
    facebase: path to similar images of `handle`.
  '''
  def __init__(self, config, stage: Literal["train", "test"]):
    print("initializing control dataset")
    if stage == "train":
      config = config.train
    elif stage == "test":
      config = config.test
    else:
      raise NotImplementedError("dataset only support train and test")
    self.ori_type: Literal["LFW", "FFHQ"] = config.ori_type
    ori_path: str = config.ori_path
    facebase: str = config.facebase
    img_size: int = getattr(config, "image_size", 128)
    self.transform = transforms.Compose([
      transforms.Resize((img_size, img_size)),
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
    ])
    if self.ori_type == "FFHQ":
      ds = FFHQ(ori_path, self.transform)
    elif self.ori_type == "LFW":
      ds = LFW(ori_path, self.transform)
    else:
      raise NotImplementedError("only support LFW and FFHQ target")
    self.target = ds
    self.base_images = sorted(
      get_image_paths_from_dir(facebase),
      key= lambda x: int(x.split(".")[0].split("_")[-1])
    )
    
    assert len(self.target) == len(self.base_images), "source and target must have the same amount"
    print(f"successfully {stage} set")
  
  def __getitem__(self, index):
    target_img, feature, label = self.target.__getitem__(index)
    base = Image.open(self.base_images[index])
    base = self.transform(base)
    return (
      (target_img, f"{self.ori_type}_{index:06}"), # x
      (base, f"{self.ori_type}_similar_{index:06}"), # x_cond
      feature) # control
  
  def __len__(self):
    return self.target.__len__()