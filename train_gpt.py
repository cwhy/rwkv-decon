from __future__ import annotations

import os
from functools import partial
from typing import Callable

import jax.numpy as jnp
import optax
import wandb
from optax import softmax_cross_entropy_with_integer_labels

import gpt
from clean_frame import batch_fy
from clean_frame_utils import Arr, init_weight_module
from custom_dataset import load_jax_cached
from gpt import Gpt
from jax_init_utils import infinite_safe_keys
from train_utils import BatchConfig, TrainConfig, TrainState, BatchType

# need to install the updated version for optax:
# pip install git+https://github.com/deepmind/optax.git

os.environ['JAX_LOG_COMPILES'] = '1'


def gpt_loss_batch(forward: Callable[[Gpt.Weights, Arr], Arr], w: Gpt.Weights, batch: BatchType) -> Arr:
    inputs, labels = batch
    logits = batch_fy(forward)(w, jnp.array(inputs))
    return softmax_cross_entropy_with_integer_labels(logits, jnp.array(labels)).mean()


dataset = "play"
encoded_jax, encode, decode, vocab_size_ = load_jax_cached(dataset=dataset)
n = int(len(encoded_jax) * 0.9)
train_data = encoded_jax[:n]
valid_data = encoded_jax[n:]

key_gen = infinite_safe_keys(0)

adam_params = {
    'learning_rate': 1e-4,
    'beta1': 0.9,
    'beta2': 0.999,
    'eps': 1e-8,
}
lion_params = {
    'learning_rate': 1e-4,
    'beta1': 0.9,
    'beta2': 0.99,
}
train_params = {
    'eval_iters': 200,
    'eval_interval': 2000,
    'max_iters': 1000000,
    # 'adam': adam_params,
    'lion': lion_params,
    # 'adamw': adam_params,
    'optimizer': 'lion',
}

experimental_params = {
    'eps': 1e-5,
    'n_tokens': vocab_size_,
    'n_channels': 768,
    'n_heads': 12,
    'n_blocks': 12,

    'batch_size': 8,
    'block_size': 128,
    'train': train_params
}


max_iters = experimental_params['train']['max_iters']
eval_interval = experimental_params['train']['eval_interval']
eval_iters = experimental_params['train']['eval_iters']
batch_config_ = BatchConfig(block_size=experimental_params['block_size'],
                            batch_size=experimental_params['batch_size'])

# dataset = "english"

gpt_config_ = Gpt.Config(eps=experimental_params['eps'],
                         n_channels=experimental_params['n_channels'],
                         n_heads=experimental_params['n_heads'],
                         # n_seq='dynamic',
                         n_seq=batch_config_.block_size,
                         max_seq_len=batch_config_.block_size,
                         n_blocks=experimental_params['n_blocks'],
                         n_tokens=vocab_size_).fill()


init_weights_ = gpt_config_.weights_check(init_weight_module(gpt_config_, next(key_gen)))
init_weights_['positional_encoding'] = gpt.get_positional_encoding(experimental_params['block_size'],
                                                                   experimental_params['n_channels'])

if experimental_params['train']['optimizer'] == 'adam':
    adam_config = experimental_params['train']['adam']
    optimizer_ = optax.adam(learning_rate=adam_config['learning_rate'],
                            b1=adam_config['beta1'],
                            b2=adam_config['beta2'],
                            eps=adam_config['eps'])
elif experimental_params['train']['optimizer'] == 'lion':
    lion_config = experimental_params['train']['lion']
    optimizer_ = optax.lion(learning_rate=lion_config['learning_rate'],
                            b1=lion_config['beta1'],
                            b2=lion_config['beta2'])
elif experimental_params['train']['optimizer'] == 'adamw':
    adamw_config = experimental_params['train']['adamw']
    optimizer_ = optax.adamw(learning_rate=adamw_config['learning_rate'],
                             b1=adamw_config['beta1'],
                             b2=adamw_config['beta2'],
                             eps=adamw_config['eps'])
else:
    raise ValueError(f"optimizer {experimental_params['train']['optimizer']} not supported")

# noinspection PyArgumentList
# cuz it's a NamedTuple
train_config_ = TrainConfig(model=gpt_config_.make(),
                            loss_fn_in=gpt_loss_batch,
                            optimiser=optimizer_)
# noinspection PyArgumentList
# cuz it's a NamedTuple
train_state_: TrainState[Gpt.Weights] = TrainState(weights=init_weights_,
                                                   opt_state=optimizer_.init(init_weights_))

wandb.init(
    project="inside-transformer",
    config=experimental_params,
)
keys_ = next(key_gen).split(max_iters)
for step in range(max_iters):
    batch_ = batch_config_.sample(train_data, keys_[step])
    if step % eval_interval == 0:
        loss = train_config_.loss_fn(train_state_.weights, batch_)
        print(f"===step {step} is an eval step===")
        print(f"before step {step}, batch loss {loss}")

    train_state_ = train_config_.train1(train_state_, batch_)
    if step % eval_interval == 0:
        loss = train_config_.loss_fn(train_state_.weights, batch_)
        print(f"after step {step}, batch loss {loss}")
        results = train_config_.estimate_loss(eval_iters, key_gen, train_state_, batch_config_,
                                    {'train': train_data, 'val': valid_data})
        # generate_f = jax.jit(partial(dynamic_model.f, train_state_.weights))
        # generated = gpt.generate(generate_f, [0], n_tokens_to_generate=10,
        #                          max_len=batch_config_.block_size)
        generate_f = partial(train_config_.model.f, train_state_.weights)
        # TODO fix generation/ add temperature
        generated = gpt.generate_static(generate_f, [0],
                                        n_tokens_to_generate=batch_config_.block_size - 1,
                                        max_len=batch_config_.block_size)
        wandb.log({"train_loss": results['train'],
                   "validation_loss": results['val'],
                   "batch_loss": loss,
                   "n_tokens_trained": step * batch_config_.batch_size * batch_config_.block_size,
                   'generated': wandb.Html(f"{decode(generated)}")})
        print(decode(generated), flush=True)

wandb.finish()
# TODO: add trainable weights