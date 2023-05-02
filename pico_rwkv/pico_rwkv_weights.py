from typing import Callable, TypeVar, Iterable, Optional

from copy_init.weights import WeightConfig, get_weights_mask
from labels import Labels
from picojax.jax_utils import Arr, WeightsTree
from jax import numpy as np


def parse_rwkv_weight(keys: Iterable[str], get_tensor: Callable[[str], Arr], trim: bool=False) -> WeightsTree:
    w = {}
    for k in keys:
        parts = k.split('.')
        last = parts.pop()
        current_ = w
        for p in parts:
            if p.isdigit():
                p = int(p)
            if p not in current_:
                current_[p] = {}
            current_ = current_[p]
        current_[last] = get_tensor(k)

    for i in w['blocks'].keys():
        att = w['blocks'][i]['att']
        ffn = w['blocks'][i]['ffn']

        for m in att, ffn:
            for k in ('key', 'value', 'receptance'):
                if k in m:
                    m[k[0] + 'w'] = m[k]['weight']
                    if trim:
                        del m[k]['weight']
    return w


def get_masks_to_train(train_tags: Optional[list[Labels]], info: dict[str, WeightConfig], trim:bool=False) -> WeightsTree:
    to_train = get_weights_mask(train_tags, info)
    mask_raw = {}
    for k, train in to_train.items():
        if train:
            mask_raw[k] = np.ones(info[k].shape, dtype=bool)
        else:
            mask_raw[k] = np.zeros(info[k].shape, dtype=bool)

    masks = parse_rwkv_weight(mask_raw.keys(), lambda k: mask_raw[k], trim)
    return masks
