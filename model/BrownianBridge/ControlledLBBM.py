import torch
import torch.nn.functional as F

from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.cldm.cldm import ControlNet, ControlledUnetModel
from model.BrownianBridge.base.util import instantiate_from_config
from model.utils import default

class ControlledLatentBrownianBridgeModel(LatentBrownianBridgeModel):
  def __init__(self, model_config):
    super().__init__(model_config)
    model_params = model_config.BB.params
    self.denoise_fn = ControlledUnetModel(**vars(model_params.UNetParams))
    self.control_model: ControlNet = instantiate_from_config(model_params.ControlNetParams)
    
  def forward(self, x, y, context=None, condition=None):
    with torch.no_grad():
      x = self.encode(x, cond=False)
      y = self.encode(y, cond=True)
    if self.condition_key == "nocond":
      context = None
    else:
      context = y if context is None else context
    b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
    assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
    t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
    return self.p_losses(x0=x, y=y, context=context, t=t, control=condition)
    
  def p_losses(self, x0, y, context, t, noise = None, control=None):
    b, c, h, w = x0.shape
    noise = default(noise, lambda: torch.randn_like(x0))
    
    x_t, objective = self.q_sample(x0, y, t, noise)
    control = self.control_model.forward(
      x_t, hint=control, 
      timesteps=t, 
      context=context
    )
    objective_recon = self.denoise_fn.forward(
      x_t, timesteps=t, 
      context=context, 
      control=control, 
      only_mid_control=False
    )

    if self.loss_type == 'l1':
        recloss = (objective - objective_recon).abs().mean()
    elif self.loss_type == 'l2':
        recloss = F.mse_loss(objective, objective_recon)
    else:
        raise NotImplementedError()

    x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
    log_dict = {
        "loss": recloss,
        "x0_recon": x0_recon
    }
    
    return recloss, log_dict