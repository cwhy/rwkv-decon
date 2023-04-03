from __future__ import annotations
from functools import partial
from typing import Callable, TypeVar

import jax
import optax
from jax import Array
Arr = Array

Weights = TypeVar('Weights')
Batch = TypeVar('Batch')


@partial(jax.jit, static_argnums=(0, 1))
def jax_calc_updates(
        optimizer: optax.GradientTransformation,
        loss_fn: Callable[[Batch], Arr],
        weights: Weights,
        batch: Batch,
        opt_state: optax.OptState
) -> tuple[Weights, optax.OptState]:
    grads = jax.grad(loss_fn)(weights, batch)
    updates, opt_state = optimizer.update(grads, opt_state, weights)
    return optax.apply_updates(weights, updates), opt_state
