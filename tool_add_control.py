import sys
import os
import yaml
input_path = "pretrained/CelebAMaskHQ-f4.pth"
output_path = "pretrained/control_CelebAMaskHQ-f4.pth"
config_path='configs/CelebAMaskHQ-f4.yaml'

assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
from utils import dict2namespace
from model.BrownianBridge.ControlledLBBM import ControlledLatentBrownianBridgeModel


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]
with open(config_path) as cf:
    config = yaml.load(cf, Loader=yaml.FullLoader)
config = dict2namespace(config)
model_conf = config.model

model = ControlledLatentBrownianBridgeModel(model_conf)

pretrained_weights = torch.load(input_path)
model_weights = pretrained_weights["model"]

scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_model')
    if is_control:
        copy_k = 'denoise_fn' + name
    else:
        copy_k = k
    if copy_k in model_weights:
        target_dict[k] = model_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

model.load_state_dict(target_dict, strict=True)
pretrained_weights['model'] = model.state_dict()
pretrained_weights['step'] = 0
pretrained_weights['epoch'] = 0
torch.save(pretrained_weights, output_path)
print('Done.')
