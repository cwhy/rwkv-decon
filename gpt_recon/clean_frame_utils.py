# %%
from __future__ import annotations

from abc import abstractmethod
from typing import NamedTuple, TypeVar, List, Union, Protocol, Mapping, cast, Callable

import jax.numpy as jnp
from typing_extensions import runtime

from picojax.jax_utils import Arr, WeightsTree
from picojax.random_utils import kaiming_init, embedding_init, normal_init, dropout_gen, SafeKey, ArrayGen


class WeightConfig(NamedTuple):
    # from in to out
    save_name: str
    shape: tuple[int, ...]
    init: Union[ArrayGen, int, float] = "kaiming"
    scale: float = 1

    def make(self, rng_key: SafeKey) -> Arr:
        if isinstance(self.init, int) or isinstance(self.init, float):
            return jnp.full(self.shape, float(self.init))
        elif self.init == 'kaiming':
            return kaiming_init(rng_key, self.scale, self.shape)
        elif self.init == 'embedding':
            return embedding_init(rng_key, self.scale, self.shape)
        elif self.init == 'normal':
            return normal_init(rng_key, self.scale, self.shape)
        elif self.init == 'dropout':
            return dropout_gen(rng_key, self.scale, self.shape)
        else:
            raise NotImplementedError("unsupported init type")


W_co = TypeVar('W_co', covariant=True)


@runtime
class ModuleConfig(Protocol[W_co]):
    @property
    @abstractmethod
    def save_name(self) -> str:
        ...

    @property
    @abstractmethod
    def weights(self) -> WeightConfigDict:
        ...

    @property
    @abstractmethod
    def parts(self) -> PartsDict:
        ...

    def fill(self) -> ModuleConfig:
        ...

    def make(self) -> Module:
        ...

    def weights_check(self, weights: WeightsTree) -> W_co:
        ...


W = TypeVar('W', contravariant=True)


@runtime
class Module(Protocol[W]):

    @abstractmethod
    def f(self, w: W, x: Arr) -> Arr:
        ...


T = TypeVar('T')


def check_config(config: NamedTuple) -> None:
    for k, v in config._asdict().items():
        if v is None:
            raise ValueError(f"Missing config '{k}' in {config.__class__}")


WeightConfigTree = Mapping[str, Union[WeightConfig, "WeightConfigTree", list["WeightConfigTree"]]]

WeightConfigDict = dict[str, WeightConfig]
PartsDict = dict[str, Union[ModuleConfig, List[ModuleConfig]]]


def config_weights_check(config: ModuleConfig, weights: WeightsTree) -> WeightsTree:
    try:
        checked_w: WeightsTree = {}
        assert isinstance(weights, dict), f"weights for {config.save_name} module is not a dict: {type(weights)}"
        for name, part_config in config.parts.items():
            if name not in weights:
                raise ValueError(f"Missing weight {name}")
            w = weights[name]
            if isinstance(part_config, ModuleConfig):
                if not isinstance(w, dict):
                    raise ValueError(f"weights for {config.save_name} module is not a dict: {type(w)}")
                checked_w[name] = config_weights_check(part_config, w)
            else:
                err_msg = f"Config {config.save_name} should be a ModuleConfig or a non-empty list of ModuleConfigs, got {part_config}"
                assert isinstance(part_config, list), err_msg
                assert len(part_config) > 0, err_msg
                if len(part_config) != len(w):
                    raise ValueError(
                        f"Wrong number of weights in list {name}: weight {len(w)} != config {len(part_config)}")
                assert isinstance(part_config[0], ModuleConfig), err_msg
                if not isinstance(w, list):
                    raise ValueError(f"weights for {name} module is not a list: {type(w)}")
                else:
                    checked_w[name] = [config_weights_check(part, cast(WeightsTree, w[i]))
                                       for i, part in enumerate(part_config)]
        checked_w.update(weights_check(config.weights, weights))
        return checked_w
    except ValueError as e:
        raise ValueError(f"Config Weights check failed for {config.save_name}") from e


def weights_check(weights_config: WeightConfigDict, weights: WeightsTree) -> WeightsTree:
    try:
        w: WeightsTree = {}
        for name, config in weights_config.items():
            assert isinstance(config, WeightConfig)
            w[name] = weight_check(config, name, weights)
        return w
    except ValueError as e:
        raise ValueError(f"Weights check failed for {weights_config}") from e


def squeeze_side(s: tuple[int, ...]) -> tuple[int, ...]:
    assert len(s) > 0, "Empty shape cannot be squeezed"
    if len(s) == 1:
        return s
    if s[0] == 1:
        return squeeze_side(s[1:])
    if s[-1] == 1:
        return squeeze_side(s[:-1])
    return s


def weight_check(config: WeightConfig, name: str, weights: WeightsTree) -> Arr:
    if name not in weights:
        raise ValueError(f"Missing weight {name}")
    w = weights[name]
    if not isinstance(w, Arr):
        raise ValueError(f"weight {name} is not an array: {type(w)}")
    if squeeze_side(w.shape) != squeeze_side(config.shape):
        raise ValueError(f"Shape for weight {name} does not match: {w.shape} != {config.shape}")
    else:
        return w.reshape(config.shape)


def load_config(config: ModuleConfig, weights_getter: Callable[[str], Arr], prefix: str = "") -> WeightsTree:
    if len(config.save_name) == 0:
        name_prefix = prefix
    else:
        name_prefix = f"{prefix}{config.save_name}."
    weights: WeightsTree = {}
    for name, part_config in config.parts.items():
        if isinstance(part_config, ModuleConfig):
            weights[name] = load_config(part_config, weights_getter, name_prefix)
        else:
            err_msg = f"Config {config.save_name} should be a ModuleConfig or a non-empty list of ModuleConfigs"
            assert isinstance(part_config, list), err_msg
            assert len(part_config) > 0, err_msg
            assert isinstance(part_config[0], ModuleConfig), err_msg
            weights[name] = [load_config(part, weights_getter, f"{name_prefix}{i}.")
                             for i, part in enumerate(part_config)]
    weights.update(load_weights(config.weights, weights_getter, name_prefix))
    return weights


def load_weights(weight_configs: WeightConfigDict, weights_getter: Callable[[str], Arr],
                 prefix: str = "") -> WeightsTree:
    weights: WeightsTree = {}
    for name, weight_config in weight_configs.items():
        assert isinstance(weight_config,
                          WeightConfig), f"weight_config {name} is not a WeightConfig, but {type(weight_config)}"
        weights[name] = weights_getter(f"{prefix}{weight_config.save_name}")
    return weights


def init_weights(weights_config: WeightConfigDict, rng_key: SafeKey) -> WeightsTree:
    w: WeightsTree = {}
    all_keys = rng_key.split(len(weights_config))
    for key, (name, weight_config) in zip(all_keys, weights_config.items()):
        assert isinstance(weight_config, WeightConfig), f"Config {name} should be a WeightConfig, got {weight_config}"
        w[name] = weight_config.make(key)
    return w


def init_weight_module(module: ModuleConfig, rng_key: SafeKey) -> WeightsTree:
    try:
        parts_key, weights_key = rng_key.split(2)
        w = init_weight_parts(module.parts, parts_key)
        w.update(init_weights(module.weights, weights_key))
    except Exception as e:
        raise Exception(f"Failed to initialize module {module.save_name}.") from e
    return w


def init_weight_parts(parts: PartsDict, rng_key: SafeKey) -> WeightsTree:
    w: WeightsTree = {}
    all_keys = rng_key.split(len(parts))
    for key, (name, part_config) in zip(all_keys, parts.items()):
        if isinstance(part_config, ModuleConfig):
            w[name] = init_weight_module(part_config, key)
        else:
            err_msg = f"Config {name} should be a ModuleConfig or a non-empty list of ModuleConfigs, got {part_config}"
            assert isinstance(part_config, list), err_msg
            assert len(part_config) > 0, err_msg
            assert isinstance(part_config[0], ModuleConfig), err_msg
            subkeys = key.split(len(part_config))
            w[name] = [init_weight_module(part, subkey)
                       for i, (subkey, part) in enumerate(zip(subkeys, part_config))]
    return w
