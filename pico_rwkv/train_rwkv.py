from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import jax.numpy as np
import optax
import wandb
from jax import vmap
from optax import softmax_cross_entropy_with_integer_labels

import nlp_utils
from copy_init.weights import get_normal_weights_config_, init
from custom_dataset import load_jax_cached
from pico_rwkv.pico_rwkv_parallel import rwkv_net_parallel
from pico_rwkv.pico_rwkv_weights import parse_rwkv_weight
from picojax.jax_utils import WeightsTree, Arr
from picojax.random_utils import infinite_safe_keys
from picojax.train_utils import BatchConfig, TrainConfig, TrainState, BatchType

os.environ['JAX_LOG_COMPILES'] = '1'
path = Path("/Data/lm_models/rwkv")
model_name = 'RWKV-4-Pile-430M-20220808-8066'

weight_infos = get_normal_weights_config_(path, model_name)

keygen = infinite_safe_keys(0)
key = next(keygen)

init_weights_raw = init(weight_infos, rng_key=key)
init_weights_ = parse_rwkv_weight(init_weights_raw.keys(), init_weights_raw.__getitem__)



def rwkv_f(w: WeightsTree, token_array: Arr) -> Arr:
    return rwkv_net_parallel(len(token_array), token_array, **w)


def loss_f(w: WeightsTree, batch: BatchType) -> Arr:
    inputs, labels = batch
    logits = vmap(rwkv_f, in_axes=(None, 0), out_axes=0)(w, np.array(inputs))
    return softmax_cross_entropy_with_integer_labels(logits, np.array(labels)).mean()


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
    'beta1': 0.95,
    'beta2': 0.98,
    'weight_decay': 0.01
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
    # 'n_channels': 768,
    # 'n_blocks': 12,

    'batch_size': 8,
    'block_size': 128,
    'train': train_params,
    'model': "rwkv"
}

max_iters = experimental_params['train']['max_iters']
eval_interval = experimental_params['train']['eval_interval']
eval_iters = experimental_params['train']['eval_iters']
batch_config_ = BatchConfig(block_size=experimental_params['block_size'],
                            batch_size=experimental_params['batch_size'])


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
                            b2=lion_config['beta2'],
                            weight_decay=lion_config['weight_decay'])
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
train_config_ = TrainConfig(loss_fn=loss_f,
                            optimiser=optimizer_)
# noinspection PyArgumentList
# cuz it's a NamedTuple
train_state_: TrainState = TrainState(weights=init_weights_,
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
        # generate_f = jax.jit(partial(dynamic_model_f, train_state_.weights))
        # generated = gpt.generate(generate_f, [0], n_tokens_to_generate=10,
        #                          max_len=batch_config_.block_size)
        generate_f = partial(rwkv_f, train_state_.weights)
        # TODO fix generation/ add temperature
        generated = nlp_utils.generate_static(generate_f, [0],
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
# TODO: add weight decay
