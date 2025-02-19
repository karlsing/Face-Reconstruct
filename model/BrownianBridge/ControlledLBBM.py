import torch
import torch.nn.functional as F

from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.base.util import instantiate_from_config
from model.cldm.cldm import ControlNet, ControlledUnetModel
from model.utils import default, exists, extract
from tqdm.auto import tqdm

class ControlledLatentBrownianBridgeModel(LatentBrownianBridgeModel):
  def __init__(self, model_config):
    super().__init__(model_config)
    model_params = model_config.BB.params
    self.denoise_fn = ControlledUnetModel(**vars(model_params.UNetParams))
    self.control_model: ControlNet = instantiate_from_config(model_params.ControlNetParams)

  def get_parameters(self):
    return self.control_model.parameters()
    
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
  
  @torch.no_grad()
  def sample(self, x_cond, clip_denoised=False, sample_mid_step=False, control=None):
    x_cond_latent = self.encode(x_cond, cond=True)
    if sample_mid_step:
      temp, one_step_temp = self.p_sample_loop(y=x_cond_latent, context=self.get_cond_stage_context(x_cond), control=control,
                                               clip_denoised=clip_denoised, sample_mid_step=sample_mid_step)
      out_samples = []
      for i in tqdm(range(len(temp)), 
                    initial=0, desc="save output sample mid steps", 
                    dynamic_ncols=True, smoothing=0.01):
        with torch.no_grad():
          out = self.decode(temp[i].detach(), cond=False)
        out_samples.append(out.to('cpu'))

      one_step_samples = []
      for i in tqdm(range(len(one_step_temp)),
                    initial=0, desc="save one step sample mid steps",
                    dynamic_ncols=True, smoothing=0.01):
        with torch.no_grad():
          out = self.decode(one_step_temp[i].detach(), cond=False)
        one_step_samples.append(out.to('cpu'))
      return out_samples, one_step_samples
    else:
      temp = self.p_sample_loop(y=x_cond_latent, context=self.get_cond_stage_context(x_cond), control=control,
                                clip_denoised=clip_denoised, sample_mid_step=sample_mid_step)
      x_latent = temp
      out = self.decode(x_latent, cond=False)
      return out
  
  @torch.no_grad()
  def p_sample_loop(self, y, context=None, clip_denoised=True, sample_mid_step=False, control=None):
    if self.condition_key == "nocond":
      context = None
    else:
      context = y if context is None else context

    if sample_mid_step:
      imgs, one_step_imgs = [y], []
      for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
        img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
        imgs.append(img)
        one_step_imgs.append(x0_recon)
      return imgs, one_step_imgs
    else:
      img = y
      for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
        img, _ = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised, control=control)
      return img
  
  @torch.no_grad()
  def p_sample(self, x_t, y, context, i, clip_denoised=False, control=None):
    b, *_, device = *x_t.shape, x_t.device
    if self.steps[i] == 0:
      t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
      control = self.control_model.forward(x_t, hint=control, timesteps=t, context=context)
      objective_recon = self.denoise_fn.forward(x_t, timesteps=t, context=context, control=control)
      x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
      if clip_denoised:
        x0_recon.clamp_(-1., 1.)
      return x0_recon, x0_recon
    else:
      t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
      n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

      objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
      x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
      if clip_denoised:
        x0_recon.clamp_(-1., 1.)

      m_t = extract(self.m_t, t, x_t.shape)
      m_nt = extract(self.m_t, n_t, x_t.shape)
      var_t = extract(self.variance_t, t, x_t.shape)
      var_nt = extract(self.variance_t, n_t, x_t.shape)
      sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
      sigma_t = torch.sqrt(sigma2_t) * self.eta

      noise = torch.randn_like(x_t)
      x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                      (x_t - (1. - m_t) * x0_recon - m_t * y)

      return x_tminus_mean + sigma_t * noise, x0_recon