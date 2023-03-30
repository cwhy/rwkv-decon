from __future__ import annotations

from typing import Tuple, Literal

from jax import Array
from jax import numpy as xp, random
from jax.random import PRNGKeyArray

ArrayGen = Literal['kaiming', 'dropout', 'embedding', 'normal']
RNGKey = PRNGKeyArray


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
