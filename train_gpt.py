from __future__ import annotations

from functools import partial
from pprint import pprint
from typing import Callable, Iterator, Iterable, TypeVar, Generic

import jax
from typing_extensions import NamedTuple

import jax.numpy as jnp
import optax
from jax import random
from optax import GradientTransformation, OptState
from optax import softmax_cross_entropy_with_integer_labels

import gpt
from clean_frame import batch_fy
from clean_frame_train_utils import jax_calc_updates
from clean_frame_utils import Arr, config_weights_check, init_weight_module, ModuleConfig, WeightsTree, \
    Module
from custom_dataset import load_jax_cached
from gpt import GptMha, Gpt
from jax_init_utils import SafeKey, infinite_safe_keys


def go(c: ModuleConfig, x: Arr, key: SafeKey) -> Arr:
    weights = init_weight_module(c, key)
    return c.make().f(weights, x)


gpt_mha_config_ = GptMha.Config(n_channels=9,
                                n_heads=3,
                                n_seq='dynamic').fill()
pprint(gpt_mha_config_.parts)

w = init_weight_module(gpt_mha_config_, SafeKey(random.PRNGKey(0)))
# print(w)
checked = config_weights_check(gpt_mha_config_, w)
# print(checked)
print(go(gpt_mha_config_, jnp.ones((5, 9)), SafeKey(random.PRNGKey(0))).shape)




class BatchConfig(NamedTuple):
    block_size: int
    batch_size: int

    def sample(self, data: Arr, rng_key: SafeKey) -> tuple[Iterable[int], Iterable[int]]:
        ix = random.randint(key=rng_key.get(), minval=0, maxval=len(data) - self.block_size, shape=(self.batch_size,))
        inputs_ = data[(ix[:, jnp.newaxis] + jnp.arange(self.block_size)[jnp.newaxis, :])]
        targets_ = data[(ix[:, jnp.newaxis] + jnp.arange(1, self.block_size + 1)[jnp.newaxis, :])]
        return inputs_, targets_


@partial(jax.jit, static_argnums=(0, ))
def gpt_loss_batch(gpt: Module[Gpt.Weights], w: Gpt.Weights, batch: BatchType) -> Arr:
    inputs, labels = batch
    logits = batch_fy(gpt.f)(w, jnp.array(inputs))
    return softmax_cross_entropy_with_integer_labels(logits, jnp.array(labels)).mean()


W = TypeVar('W')


class TrainState(NamedTuple, Generic[W]):
    model: Module
    loss_fn_in: Callable[[Module[W], W, BatchType], Arr]
    optimiser: GradientTransformation
    weights: W
    opt_state: OptState

    def f(self, x: Arr) -> Arr:
        return jax.jit(self.model.f)(self.weights, x)

    def loss_fn(self, weights: W, batch: BatchType) -> Arr:
        return self.loss_fn_in(self.model, weights, batch)

    def loss(self, batch: BatchType) -> Arr:
        return self.loss_fn_in(self.model, self.weights, batch)

    def train1(self, batch: BatchType) -> TrainState:
        weights, opt_state = jax_calc_updates(self.optimiser,
                                              self.loss_fn,
                                              self.weights,
                                              batch,
                                              self.opt_state)
        return self._replace(weights=weights, opt_state=opt_state)


BatchType = tuple[Iterable[int], Iterable[int]]


# %%
def estimate_loss(eval_iters: int,
                  rng_key_gen: Iterator[SafeKey],
                  train_state: TrainState,
                  batch_config: BatchConfig,
                  data: dict[str, Arr]) -> None:
    for split in 'train', 'val':
        total_eval_loss = 0
        for key in next(rng_key_gen).split(eval_iters):
            eval_batch = batch_config.sample(data[split], key)
            total_eval_loss += train_state.loss(eval_batch).item()
        print(f"Estimated {split} loss: {total_eval_loss / eval_iters}")


key_gen = infinite_safe_keys(0)
max_iters = 5
eval_interval = 10
learning_rate_ = 1e-4
eval_iters = 200
batch_config_ = BatchConfig(block_size=16, batch_size=4)

dataset = "english"
# dataset = "play"
encoded_jax, encode, decode, vocab_size_ = load_jax_cached(dataset=dataset)
n = int(len(encoded_jax) * 0.9)
train_data = encoded_jax[:n]
valid_data = encoded_jax[n:]

gpt_config_ = Gpt.Config(eps=1e-5,
                         n_channels=32,
                         n_heads=4,
                         n_seq='dynamic',
                         max_seq_len=batch_config_.block_size,
                         n_blocks=4,
                         n_tokens=vocab_size_).fill()
# result = go(gpt_config_, jnp.ones((1024,), dtype=jnp.int32), next(key_gen))
# print(result.shape)

init_weights_ = gpt_config_.weights_check(init_weight_module(gpt_config_, next(key_gen)))
optimizer_ = optax.adam(learning_rate=learning_rate_)
train_state_: TrainState[Gpt.Weights] = TrainState(model=gpt_config_.make(),
                                                   loss_fn_in=gpt_loss_batch,
                                                   optimiser=optimizer_,
                                                   weights=init_weights_,
                                                   opt_state=optimizer_.init(init_weights_))

with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):

    keys_ = next(key_gen).split(max_iters)
    for step in range(max_iters):
        print(step)
        batch_ = batch_config_.sample(train_data, keys_[step])
        if step % eval_interval == 0:
            loss = train_state_.loss(batch_)
            print(f"===step {step} is an eval step===")
            print(f"before step {step}, batch loss {loss}")

        train_state_ = train_state_.train1(batch_)
        if step % eval_interval == 0:
            loss = train_state_.loss(batch_)
            print(f"after step {step}, batch loss {loss}")
            estimate_loss(eval_iters, key_gen, train_state_, batch_config_, {'train': train_data, 'val': valid_data})
            generated = gpt.generate(train_state_.f, [0], n_tokens_to_generate=10, max_len=batch_config_.block_size)
            print(decode(generated), flush=True)

# TODO: fix profiling