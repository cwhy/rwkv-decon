# %%
from __future__ import annotations

from pathlib import Path

import jax.numpy as np
from jax.tree_util import tree_flatten

from copy_init.weights import get_normal_weights_config_, init, save_pytree_, load_pytree_
from pico_rwkv.pico_rwkv_weights import parse_rwkv_weight
from picojax.jax_utils import WeightsTree
from picojax.random_utils import infinite_safe_keys

model_path = Path("/Data/lm_models/rwkv")
# model_name = 'RWKV-4-Pile-430M-20220808-8066'
model_name = 'RWKV-4-Pile-169M-20220807-8023'

weight_infos = get_normal_weights_config_(model_path, model_name)
keygen = infinite_safe_keys(0)
key = next(keygen)
init_weights_raw = init(weight_infos, rng_key=key)
init_weights_: WeightsTree = parse_rwkv_weight(init_weights_raw.keys(), init_weights_raw.__getitem__, trim=True)
_, tree_struct = tree_flatten(init_weights_)
f = save_pytree_(init_weights_, ".", model_name)
print(f)
ww = load_pytree_(tree_struct, ".", model_name)


def nested_dict_equal(a, b):
    """
    Check if two nested dictionaries of NumPy arrays and lists are equal.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        for key in a.keys():
            if not nested_dict_equal(a[key], b[key]):
                print("kd", key)
                print(a[key].shape, b[key].shape)
                return False
        return True
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if not nested_dict_equal(a[i], b[i]):
                return False
        return True
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    else:
        return a == b


print(nested_dict_equal(init_weights_, ww))
