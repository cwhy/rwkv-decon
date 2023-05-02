# %%
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol, Literal, Optional, cast

import jax.numpy as np
from jax import random
from jax.tree_util import tree_flatten, tree_unflatten, PyTreeDef
from pydantic import ValidationError
from pydantic.dataclasses import dataclass
from pydantic.json import pydantic_encoder
from safetensors import safe_open
from safetensors.flax import save_file

from labels import Labels, L
from picojax.jax_utils import Arr, WeightsTree
from picojax.random_utils import SafeKey

WeightConfigType = Literal['normal', 'zero']


class WeightConfig(Protocol):
    name: str
    tags: Labels
    shape: tuple[int, ...]

    def init(self, rng_key: SafeKey) -> Arr:
        ...

    @classmethod
    def from_arr(cls, name: str, tags: Labels, arr: Arr) -> WeightConfig:
        ...


@dataclass
class NormalWeight:
    name: str
    tags: Labels
    shape: tuple[int, ...]
    mean: float = 0
    scale: float = 1

    def init(self, rng_key: SafeKey) -> Arr:
        return random.normal(rng_key.get(), self.shape) * self.scale + self.mean

    @classmethod
    def from_arr(cls, name: str, tags: Labels, arr: Arr) -> NormalWeight:
        return cls(name=name, tags=tags, shape=arr.shape, mean=arr.mean().item(), scale=arr.std().item())


@dataclass
class ZeroWeight:
    name: str
    tags: Labels
    shape: tuple[int, ...]

    def init(self, rng_key: SafeKey) -> Arr:
        return np.zeros(self.shape)

    @classmethod
    def from_arr(cls, name: str, tags: Labels, arr: Arr) -> ZeroWeight:
        return cls(name=name, tags=tags, shape=arr.shape)


def get_weight_config(weight_config_type: WeightConfigType, key: str, arr: Arr, tags: Labels) -> WeightConfig:
    if weight_config_type == 'normal':
        return NormalWeight.from_arr(name=key, tags=tags, arr=arr)
    elif weight_config_type == 'zero':
        return ZeroWeight.from_arr(name=key, tags=tags, arr=arr)
    else:
        raise ValueError(f"weight_config_type must be 'normal' or 'zero', not {weight_config_type}")


def get_normal_weights_config_(path: Path, model_name: str,
                               non_normal_weight_tags: Optional[dict[str, WeightConfigType]] = None) -> dict[
    str, WeightConfig]:
    try:
        return load_weight_configs_(path, model_name)
    except FileNotFoundError:
        weight_infos: dict[str, WeightConfig] = {}
        with safe_open(path / f"{model_name}.safetensors", framework="flax", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                # weight_infos[key] = NormalWeight.from_tensor(name=key, tags=L(*key.split('.')), arr=t)
                tags = L(*key.split('.'))
                weight_config_type = 'normal'
                if non_normal_weight_tags is not None:
                    for tag, _type in non_normal_weight_tags.items():
                        if (tag in tags.tags) and (_type in ['normal', 'zero']):
                            weight_config_type = _type
                weight_config_type = cast(WeightConfigType, weight_config_type)
                weight_infos[key] = get_weight_config(weight_config_type, key, t, tags)

        with open(path / f"{model_name}.json", 'w') as f:
            json.dump(weight_infos, f, indent=2, default=pydantic_encoder)
        return weight_infos


def load_weight_configs_(path: Path, model_name: str) -> dict[str, WeightConfig]:
    with open(path / f"{model_name}.json") as f:
        weight_info_dict = json.load(f)
    weight_infos: dict[str, WeightConfig] = {}
    for key, value in weight_info_dict.items():
        try:
            weight_infos[key] = NormalWeight(**value)

        except ValidationError as e:
            print(e)
    return weight_infos


def init(w_infos: dict[str, WeightConfig], rng_key: SafeKey) -> dict[str, Arr]:
    rng_keys = rng_key.split(len(w_infos))
    return {key: w_info.init(rng_key)
            for key, w_info, rng_key in zip(w_infos.keys(), w_infos.values(), rng_keys)}


def get_weights_mask(whitelist: list[Labels], w_infos: dict[str, WeightConfig]) -> dict[str, bool]:
    def is_in_whitelist(w_info: WeightConfig) -> bool:
        for tag in whitelist:
            if w_info.tags.covers(tag):
                return True
        return False

    return {key: is_in_whitelist(w_info) for key, w_info in w_infos.items()}


def save_pytree_(w: WeightsTree, checkpoint_path: str, model_name: str) -> str:
    tensors, _ = tree_flatten(w)
    file_path = os.path.join(checkpoint_path, f"{model_name}.safetensors")
    n_tensors = len(tensors)
    save_file({str(i).zfill(len(str(n_tensors))): t for i, t in enumerate(tensors)}, file_path)
    return file_path


def load_pytree_(t: PyTreeDef, checkpoint_path: str, model_name: str) -> WeightsTree:
    file_path = os.path.join(checkpoint_path, f"{model_name}.safetensors")
    with safe_open(file_path, framework="flax") as f:
        v = iter(f.get_tensor(k) for k in f.keys())
        return tree_unflatten(t, v)
