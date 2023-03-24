# %%
from __future__ import annotations

from abc import abstractmethod
from typing import NamedTuple, TypeVar, List, Union, Literal, Protocol, Mapping, cast, Callable

import jax
from typing_extensions import runtime

ArrayGen = Literal['kaiming', 'dropout', 'embedding', 'normal']
Arr = jax.Array


class WeightConfig(NamedTuple):
    # from in to out
    name: str
    shape: tuple[int, ...]
    init: Union[ArrayGen, int, float] = "kaiming"
    scale: float = 1


class NNModule(Protocol):
    def __init__(self, config: NamedTuple):
        ...

    def f(self, w: dict, x: Arr) -> Arr:
        ...


@runtime
class ModuleConfig(Protocol):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def weights(self) -> WeightConfigDict:
        ...

    @property
    @abstractmethod
    def parts(self) -> PartsDict:
        ...


T = TypeVar('T')


def check_config(config: NamedTuple) -> None:
    for k, v in config._asdict().items():
        if v is None:
            raise ValueError(f"Missing config '{k}' in {config.__class__}")


WeightConfigTree = Mapping[str, Union[WeightConfig, "WeightConfigTree", list["WeightConfigTree"]]]
WeightsTree = dict[str, Union[Arr, "WeightsTree", list[Union["WeightsTree", Arr]]]]

WeightConfigDict = dict[str, WeightConfig]
PartsDict = dict[str, Union[ModuleConfig, List[ModuleConfig]]]


def config_weights_check(config: ModuleConfig, weights: WeightsTree) -> WeightsTree:
    try:
        checked_w: WeightsTree = {}
        assert isinstance(weights, dict), f"weights for {config.name} module is not a dict: {type(weights)}"
        for name, part_config in config.parts.items():
            if name not in weights:
                raise ValueError(f"Missing weight {name}")
            w = weights[name]
            if isinstance(part_config, ModuleConfig):
                if not isinstance(w, dict):
                    raise ValueError(f"weights for {config.name} module is not a dict: {type(w)}")
                checked_w[name] = config_weights_check(part_config, w)
            else:
                err_msg = f"Config {config.name} should be a ModuleConfig or a non-empty list of ModuleConfigs, got {part_config}"
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
        raise ValueError(f"Config Weights check failed for {config.name}") from e


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
    if len(config.name) == 0:
        name_prefix = prefix
    else:
        name_prefix = f"{prefix}{config.name}."
    weights: WeightsTree = {}
    for name, part_config in config.parts.items():
        if isinstance(part_config, ModuleConfig):
            weights[name] = load_config(part_config, weights_getter, name_prefix)
        else:
            err_msg = f"Config {config.name} should be a ModuleConfig or a non-empty list of ModuleConfigs"
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
        weights[name] = weights_getter(f"{prefix}{weight_config.name}")
    return weights
