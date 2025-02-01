from typing import TypedDict
import torch
import torch.nn.functional as F
from torch import nn
from .vgg_perp_loss import PerceptualLoss
from .dgw_gan import DGWGAN

class ExtraLoss(nn.Module):
  class ExtraLossItem(TypedDict):
    mse: torch.Tensor
    vgg: torch.Tensor
    dg : torch.Tensor

  def __init__(self):
    super().__init__()
    # self.dg = DGWGAN(in_dim=args.nc,dim=args.img_size).to(device)
    self.vgg_loss = PerceptualLoss()

  
  def forward(self, real: torch.Tensor, fake: torch.Tensor) -> ExtraLossItem:
    mse_loss = F.mse_loss(real, fake)
    vgg_loss = self.vgg_loss.forward(real, fake)
    # TODO: GAN loss    
    return {
      "mse": mse_loss, 
      "vgg": vgg_loss, 
      "dg" : None
    }
    
    