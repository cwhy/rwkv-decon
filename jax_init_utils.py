from __future__ import annotations

from typing import Tuple, Literal, Iterator

import jax
from jax import Array
from jax import numpy as xp, random
from jax.random import PRNGKeyArray

ArrayGen = Literal['kaiming', 'dropout', 'embedding', 'normal']


class SafeKey:
    """Safety wrapper for PRNG keys."""

    def __init__(self, key: PRNGKeyArray):
        self._key = key
        self._used = False

    def _assert_not_used(self) -> None:
        if self._used:
            raise RuntimeError('Random key has been used previously.')

    def get(self) -> PRNGKeyArray:
        self._assert_not_used()
        self._used = True
        return self._key

    def split(self, num_keys=2) -> Tuple['SafeKey', ...]:
        self._assert_not_used()
        self._used = True
        new_keys = jax.random.split(self._key, num_keys)
        return jax.tree_map(SafeKey, tuple(new_keys))

    def duplicate(self, num_keys=2) -> Tuple['SafeKey', ...]:
        self._assert_not_used()
        self._used = True
        return tuple(SafeKey(self._key) for _ in range(num_keys))


def infinite_safe_keys(seed: int) -> Iterator[SafeKey]:
    init_key = jax.random.PRNGKey(seed)
    while True:
        init_key, key = jax.random.split(init_key)
        yield SafeKey(key)


def dropout_gen(rng_key: SafeKey, keep_rate: float, shape: Tuple[int, ...]):
    return random.bernoulli(rng_key.get(), keep_rate, shape)


def kaiming_init(rng_key: SafeKey, sd: float, shape: Tuple[int, ...]) -> Array:
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


def embedding_init(rng_key: SafeKey, scale: float, shape: Tuple[int, ...]) -> Array:
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
    return random.normal(rng_key.get(), shape) * xp.sqrt(dim_model) * scale


def normal_init(rng_key: SafeKey, sd: float, shape: Tuple[int, ...]) -> Array:
    return random.normal(rng_key.get(), shape) * sd
