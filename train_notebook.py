from __future__ import annotations

import os
from functools import partial
from pprint import pprint
from typing import Callable

import jax.numpy as jnp
import optax
from jax import random
from optax import softmax_cross_entropy_with_integer_labels

import gpt
from clean_frame import batch_fy
from clean_frame_utils import Arr, config_weights_check, init_weight_module, ModuleConfig
from custom_dataset import load_jax_cached
from gpt import GptMha, Gpt
from jax_init_utils import SafeKey, infinite_safe_keys
from train_utils import BatchConfig, TrainConfig, TrainState, BatchType

os.environ['JAX_LOG_COMPILES'] = '1'


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


# @partial(jax.jit, static_argnums=(0,))
def gpt_loss_batch(forward: Callable[[Gpt.Weights, Arr], Arr], w: Gpt.Weights, batch: BatchType) -> Arr:
    inputs, labels = batch
    logits = batch_fy(forward)(w, jnp.array(inputs))
    return softmax_cross_entropy_with_integer_labels(logits, jnp.array(labels)).mean()


# %%


key_gen = infinite_safe_keys(0)
max_iters = 100000
eval_interval = 1000
learning_rate_ = 1e-4
eval_iters = 100
batch_config_ = BatchConfig(block_size=128, batch_size=4)

# dataset = "english"
dataset = "play"
encoded_jax, encode, decode, vocab_size_ = load_jax_cached(dataset=dataset)
n = int(len(encoded_jax) * 0.9)
train_data = encoded_jax[:n]
valid_data = encoded_jax[n:]

gpt_config_ = Gpt.Config(eps=1e-5,
                         n_channels=768,
                         n_heads=12,
                         # n_seq='dynamic',
                         n_seq=batch_config_.block_size,
                         max_seq_len=batch_config_.block_size,
                         n_blocks=12,
                         n_tokens=vocab_size_).fill()
gpt_dynamic_config_ = gpt_config_._replace(n_seq='dynamic')
dynamic_model = gpt_dynamic_config_.make()
# result = go(gpt_config_, jnp.ones((1024,), dtype=jnp.int32), next(key_gen))
# print(result.shape)

init_weights_ = gpt_config_.weights_check(init_weight_module(gpt_config_, next(key_gen)))
optimizer_ = optax.adam(learning_rate=learning_rate_)

# noinspection PyArgumentList
# cuz it's a NamedTuple
train_config_ = TrainConfig(model=gpt_config_.make(),
                            loss_fn_in=gpt_loss_batch,
                            optimiser=optimizer_)
# noinspection PyArgumentList
# cuz it's a NamedTuple
train_state_: TrainState[Gpt.Weights] = TrainState(weights=init_weights_,
                                                   opt_state=optimizer_.init(init_weights_))

# %%
# precompile for better profiling
# batch_ = batch_config_.sample(train_data, next(key_gen))
# train_state_ = train_config_.train1(train_state_, batch_)
# batch_ = batch_config_.sample(train_data, next(key_gen))
# train_state_ = train_config_.train1(train_state_, batch_)
# batch_ = batch_config_.sample(train_data, next(key_gen))
# train_state_ = train_config_.train1(train_state_, batch_)
# # print([x.device() for x in tree_flatten(train_state_.weights)[0]])
# train_config_.estimate_loss(eval_iters,
#                             key_gen,
#                             train_state_,
#                             batch_config_,
#                             {'train': train_data, 'val': valid_data})
# train_config_.estimate_loss(eval_iters,
#                             key_gen,
#                             train_state_,
#                             batch_config_,
#                             {'train': train_data, 'val': valid_data})

# %%

keys_ = next(key_gen).split(max_iters)
#with jax.profiler.trace("./jax-trace", create_perfetto_link=True):
for step in range(max_iters):
    batch_ = batch_config_.sample(train_data, keys_[step])
    if step % eval_interval == 0:
        loss = train_config_.loss_fn(train_state_.weights, batch_)
        print(f"===step {step} is an eval step===")
        print(f"before step {step}, batch loss {loss}")
        train_config_.estimate_loss(eval_iters,
                                    key_gen,
                                    train_state_,
                                    batch_config_,
                                    {'train': train_data, 'val': valid_data})

    # print(make_jaxpr(partial(jax_calc_updates, train_state_.optimiser, train_state_.loss_fn))(train_state_.weights, batch_, train_state_.opt_state))
    train_state_ = train_config_.train1(train_state_, batch_)
    if step % eval_interval == 0:
        loss = train_config_.loss_fn(train_state_.weights, batch_)
        print(f"after step {step}, batch loss {loss}")
        train_config_.estimate_loss(eval_iters, key_gen, train_state_, batch_config_,
                                    {'train': train_data, 'val': valid_data})
        # generate_f = jax.jit(partial(dynamic_model.f, train_state_.weights))
        # generated = gpt.generate(generate_f, [0], n_tokens_to_generate=10,
        #                          max_len=batch_config_.block_size)
        generate_f = partial(train_config_.model.f, train_state_.weights)
        # TODO fix generation/ add temperature
        generated = gpt.generate_static(generate_f, [0],
                                        n_tokens_to_generate=batch_config_.block_size - 1,
                                        max_len=batch_config_.block_size)
        print(decode(generated), flush=True)

