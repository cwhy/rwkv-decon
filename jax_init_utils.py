from __future__ import annotations

from abc import abstractmethod
from typing import Tuple, NamedTuple, Union, Literal, Mapping, Protocol, TypeVar, ItemsView

from jax import numpy as xp, random
from jax import Array
from jax.random import PRNGKeyArray

ArrayGen = Literal['kaiming', 'dropout', 'embedding', 'normal']
RNGKey = PRNGKeyArray


class ArrayParams(Protocol):
    # from in to out
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]: ...

    @property
    @abstractmethod
    def init(self) -> Union[ArrayGen, int, float]: ...

    @property
    @abstractmethod
    def scale(self) -> float: ...


ArrayParamMapping = Mapping[str, Union[ArrayParams, 'ArrayParamMapping']]
ArrayParamTree = Union[ArrayParamMapping, ArrayParams]
ArrayTreeMapping = Mapping[str, Union['ArrayTreeMapping', Array]]
ArrayTree = Union[ArrayTreeMapping, Array]


def get_arr(tree: ArrayTreeMapping, item: str) -> Array:
    msg = f"Trying to access array item {item} from tree {tree}"
    assert item in tree, f"{msg} but the item is not in the tree"
    arr = tree[item]
    assert isinstance(arr, xp.ndarray), f"{msg} but the result is not an array"
    return arr


def get_mapping(tree: ArrayTreeMapping, item: str) -> ArrayTreeMapping:
    msg = f"Trying to access mapping item {item} from tree {tree}"
    assert item in tree, f"{msg} but the item is not in the tree"
    arr = tree[item]
    assert not isinstance(arr, xp.ndarray), f"{msg} but the result an array"
    return arr


class WeightConfig(NamedTuple):
    # from in to out
    name: str
    shape: Tuple[int, ...]
    init: Union[ArrayGen, int, float] = "kaiming"
    scale: float = 1


def dropout_gen(rng_key: RNGKey, keep_rate: float, shape: Tuple[int, ...]):
    return random.bernoulli(rng_key, keep_rate, shape)


def kaiming_init(rng_key: RNGKey, sd: float, shape: Tuple[int, ...]) -> Array:
    """
    Generate randomly initialized weight matrix with Kaiming initalization:
    Normally distributed scaled by sqrt(2/fan_in)

    Arguments:
        :param rng_key:  random number generator key from jax
        :param sd:  standard deviation for initialization
        :param shape:  = (n_in, ..., n_out)
            where
            n_in is number of inputs to the layer
            n_out is number of outputs from the layer

    Returns:
        weight matrix of shape [n_in, n_out]
    """
    n_in = shape[0]
    return xp.sqrt(2 / n_in) * normal_init(rng_key, sd, shape)


def embedding_init(rng_key: RNGKey, scale: float, shape: Tuple[int, ...]) -> Array:
    """
    Arguments:
        :param rng_key:  random number generator key from jax
        :param scale:  standard deviation for initialization
        :param shape:  = (dict_size, ..., dim_model)
    where

    Returns:
    weight matrix of shape (dict_size, ..., dim_model)
    """
    dim_model = shape[-1]
    return random.normal(rng_key, shape) * xp.sqrt(dim_model) * scale


def normal_init(rng_key: RNGKey, sd: float, shape: Tuple[int, ...]) -> Array:
    return random.normal(rng_key, shape) * sd


def array_gen(rng_key: RNGKey, params: ArrayParams) -> Array:
    if isinstance(params.init, int) or isinstance(params.init, float):
        return xp.full(params.shape, float(params.init))
    elif params.init == 'kaiming':
        return kaiming_init(rng_key, params.scale, params.shape)
    elif params.init == 'embedding':
        return embedding_init(rng_key, params.scale, params.shape)
    elif params.init == 'normal':
        return normal_init(rng_key, params.scale, params.shape)
    elif params.init == 'dropout':
        return dropout_gen(rng_key, params.scale, params.shape)
    else:
        raise NotImplementedError("unsupported init type")


def init_weights_helper(rng_key: RNGKey, params: ArrayParamTree) -> ArrayTree:
    if isinstance(params, WeightConfig):
        new_key = random.split(rng_key, 1)[0]
        return array_gen(new_key, params)
    else:
        assert isinstance(params, dict)
        rng_keys = random.split(rng_key, len(params))
        return {
            k: init_weights_helper(rng_keys[i], v)
            for i, (k, v) in enumerate(params.items()) if v is not None
        }


def make_weights(rng_key: RNGKey, params: ArrayParamMapping) -> ArrayTreeMapping:
    rng_keys = random.split(rng_key, len(params))
    return {
        k: init_weights_helper(rng_keys[i], v)
        for i, (k, v) in enumerate(params.items()) if v is not None
    }
