from __future__ import annotations

from functools import partial
from typing import Callable, TypeVar, Optional, Union
from typing import Iterable, Generic, Iterator

import jax
import optax
from jax import random, numpy as jnp, vmap, numpy as np
from optax import GradientTransformation, OptState, softmax_cross_entropy_with_integer_labels
from typing_extensions import NamedTuple

from nlp_utils import Tokenizer
from .jax_utils import Arr, WeightsTree
from .random_utils import SafeKey

Weights = TypeVar('Weights')
Batch = TypeVar('Batch')


@partial(jax.jit, static_argnums=(0, 1), inline=True)
def jax_calc_updates(
        optimizer: optax.GradientTransformation,
        loss_fn: Callable[[Batch], Arr],
        weights: Weights,
        train_mask: Optional[WeightsTree],
        batch: Batch,
        opt_state: optax.OptState
) -> tuple[Weights, optax.OptState]:
    grads = jax.grad(loss_fn)(weights, batch)
    if train_mask is not None:
        grads = jax.tree_map(lambda x, y: x * y, grads, train_mask)
    updates, opt_state = optimizer.update(grads, opt_state, weights)
    return optax.apply_updates(weights, updates), opt_state


class BatchConfig(NamedTuple):
    block_size: int
    batch_size: int
    tokenizer: Tokenizer
    str_sampling: bool = False


    def sample_jax(self, data: Arr, rng_key: SafeKey) -> tuple[Iterable[int], Iterable[int]]:
        ix = random.randint(key=rng_key.get(), minval=0, maxval=len(data) - self.block_size, shape=(self.batch_size,))
        inputs_ = data[(ix[:, jnp.newaxis] + jnp.arange(self.block_size)[jnp.newaxis, :])]
        targets_ = data[(ix[:, jnp.newaxis] + jnp.arange(1, self.block_size + 1)[jnp.newaxis, :])]
        return inputs_, targets_

    def sample_from_str(self, data: str, rng_key: SafeKey) -> tuple[Iterable[int], Iterable[int]]:
        ix = random.randint(key=rng_key.get(), minval=0, maxval=len(data) - self.block_size, shape=(self.batch_size,))
        arr = np.stack([np.array(self.tokenizer.encode(data[i:i + self.block_size]).ids).astype(np.int32) for i in ix])
        inputs_ = arr[:, :-1]
        targets_ = arr[:, 1:]
        return inputs_, targets_

    def sample(self, data: Union[str, Arr], rng_key: SafeKey) -> tuple[Iterable[int], Iterable[int]]:
        if self.str_sampling:
            assert isinstance(data, str), 'data must be a string when str_sampling is True'
            return self.sample_from_str(data, rng_key)
        else:
            assert not isinstance(data, str), 'data must be an array when str_sampling is False'
            return self.sample_jax(data, rng_key)

W = TypeVar('W')


class TrainConfig(NamedTuple, Generic[W]):
    loss_fn: Callable[[W, BatchType], Arr]
    optimiser: GradientTransformation

    def estimate_loss(self,
                      eval_iters: int,
                      rng_key_gen: Iterator[SafeKey],
                      train_state: TrainState,
                      batch_config: BatchConfig,
                      data: dict[str, Arr]) -> dict[str, float]:

        results = {}
        for split in data.keys():
            total_eval_loss = 0
            for key in next(rng_key_gen).split(eval_iters):
                eval_batch = batch_config.sample(data[split], key)
                total_eval_loss += self.loss_fn(train_state.weights, eval_batch).item()
            results[split] = total_eval_loss / eval_iters
            print(f"Estimated {split} loss: {total_eval_loss / eval_iters}")
        return results

    def train1(self, state: TrainState, batch: BatchType) -> TrainState:
        weights, opt_state = jax_calc_updates(self.optimiser,
                                              self.loss_fn,
                                              state.weights,
                                              state.train_mask,
                                              batch,
                                              state.opt_state)
        return state.update(weights=weights, opt_state=opt_state)


class TrainState(Generic[W], NamedTuple):
    weights: W
    opt_state: OptState
    train_mask: Optional[WeightsTree] = None

    def update(self, **kwargs):
        return self._replace(**kwargs)


BatchType = tuple[Iterable[int], Iterable[int]]


def get_lm_loss(f: Callable[[WeightsTree, Arr], Arr], w: WeightsTree, batch: BatchType) -> Arr:
    inputs, labels = batch
    logits = vmap(f, in_axes=(None, 0), out_axes=0)(w, np.array(inputs))
    return softmax_cross_entropy_with_integer_labels(logits, np.array(labels)).mean()
