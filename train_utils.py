from __future__ import annotations

from typing import Iterable, TypeVar, Generic, Callable, Iterator

from jax import random, numpy as jnp
from optax._src.base import GradientTransformation, OptState
from typing_extensions import NamedTuple

from clean_frame_train_utils import jax_calc_updates
from clean_frame_utils import Arr, Module, jit_f
from jax_init_utils import SafeKey


class BatchConfig(NamedTuple):
    block_size: int
    batch_size: int

    def sample(self, data: Arr, rng_key: SafeKey) -> tuple[Iterable[int], Iterable[int]]:
        ix = random.randint(key=rng_key.get(), minval=0, maxval=len(data) - self.block_size, shape=(self.batch_size,))
        inputs_ = data[(ix[:, jnp.newaxis] + jnp.arange(self.block_size)[jnp.newaxis, :])]
        targets_ = data[(ix[:, jnp.newaxis] + jnp.arange(1, self.block_size + 1)[jnp.newaxis, :])]
        return inputs_, targets_


W = TypeVar('W')


class TrainConfig(NamedTuple, Generic[W]):
    model: Module
    loss_fn_in: Callable[[Callable[[W, Arr], Arr], W, BatchType], Arr]
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

    @jit_f
    def loss_fn(self, weights: W, batch: BatchType) -> Arr:
        return self.loss_fn_in(self.model.f, weights, batch)

    def train1(self, state: TrainState, batch: BatchType) -> TrainState:
        weights, opt_state = jax_calc_updates(self.optimiser,
                                              self.loss_fn,
                                              state.weights,
                                              batch,
                                              state.opt_state)
        return state.update(weights=weights, opt_state=opt_state)


class TrainState(Generic[W], NamedTuple):
    weights: W
    opt_state: OptState

    def update(self, **kwargs):
        return self._replace(**kwargs)


BatchType = tuple[Iterable[int], Iterable[int]]
