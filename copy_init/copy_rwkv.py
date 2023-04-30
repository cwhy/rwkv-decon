#%%
from __future__ import annotations

from pathlib import Path

from copy_init.weights import get_normal_weights_config_, init
from picojax.random_utils import infinite_safe_keys

path = Path("/Data/lm_models/rwkv")
model_name = 'RWKV-4-Pile-430M-20220808-8066'

weight_infos = get_normal_weights_config_(path, model_name)

keygen = infinite_safe_keys(0)
key = next(keygen)

w = init(weight_infos, rng_key=key)

print(w)