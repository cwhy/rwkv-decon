from __future__ import annotations

import jax
from typing import Callable, Union, TypeVar

Arr = jax.Array


def jit_f(f: Callable) -> Callable:
    return jax.jit(f, static_argnums=(0,), inline=True)


WeightsTree = dict[str, 'WeightsType']
WeightsType = Union[Arr, WeightsTree, list[Union[WeightsTree, Arr]]]

