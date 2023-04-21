from pathlib import Path

import torch
import jax
from safetensors.flax import save_file

path = Path("/Data/lm_models/rwkv")
model_name = 'RWKV-4-Pile-430M-20220808-8066'

w_raw = torch.load(path/f'{model_name}.pth', map_location='cpu')

for k in w_raw.keys():
    if '.time_' in k:
        w_raw[k] = w_raw[k].squeeze()
    if '.time_decay' in k:
        w_raw[k] = -torch.exp(w_raw[k])  # the real time decay is like e^{-e^x}
    else:
        w_raw[k] = w_raw[k]  # convert to f32 type
    w_raw[k] = jax.numpy.array(w_raw[k].float())

save_file(w_raw, path/f'{model_name}.safetensors')
