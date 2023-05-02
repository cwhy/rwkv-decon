from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Optional

import jax.numpy as np
import optax
import wandb
from jax.tree_util import tree_flatten
from tokenizers import Tokenizer

import custom_dataset
import custom_dataset_str
import nlp_utils
from copy_init.weights import get_normal_weights_config_, init, save_pytree_
from labels import Labels
from pico_rwkv.pico_rwkv_parallel import rwkv_net_parallel
from pico_rwkv.pico_rwkv_rnn import rwkv_net_rnn
from pico_rwkv.pico_rwkv_weights import parse_rwkv_weight
from picojax.jax_utils import WeightsTree, Arr
from picojax.random_utils import infinite_safe_keys
from picojax.train_utils import BatchConfig, TrainConfig, TrainState, get_lm_loss
from python_utils import num_short_form

os.environ['JAX_LOG_COMPILES'] = '1'
model_path = Path("/Data/lm_models/rwkv")
# model_name = 'RWKV-4-Pile-430M-20220808-8066'
model_name = 'RWKV-4-Pile-169M-20220807-8023'

weight_infos = get_normal_weights_config_(model_path, model_name)

keygen = infinite_safe_keys(0)
key = next(keygen)

init_weights_raw = init(weight_infos, rng_key=key)
all_weight_names = list(init_weights_raw.keys())

# randomly initialize weights
init_weights_: WeightsTree = parse_rwkv_weight(init_weights_raw.keys(), init_weights_raw.__getitem__, trim=True)
## load weights instead of randomly initialize
# with safe_open(model_path / f"{model_name}.safetensors", framework="flax", device="cpu") as f:
#     init_weights_ = parse_rwkv_weight(f.keys(), f.get_tensor)

n_channels = init_weights_['emb']['weight'].shape[1]  # type: ignore
_, tree_struct = tree_flatten(init_weights_)

train_tags: Optional[list[Labels]] = None
weight_mask = None


# train_tags = [Labels.from_strs('ffn', 'key'), Labels.from_strs('ffn', 'value')]
# weight_mask = get_masks_to_train(train_tags, weight_infos, trim=True)


# %%
def rwkv_f(w: WeightsTree, token_array: Arr) -> Arr:
    return rwkv_net_parallel(len(token_array), token_array, **w)


def rwkv_rnn(w: WeightsTree, token_array: Arr, state: Arr) -> tuple[Arr, Arr]:
    return rwkv_net_rnn(token_array, state, **w)

data_path = Path("/Data/nlp/")

dataset = "play"
train_str, tokenizer = custom_dataset.load_jax_cached_tokenizer(data_path, dataset)
init_weights_['emb']['weight'] = init_weights_['emb']['weight'][:tokenizer.get_vocab_size(), :]  # type: ignore
init_weights_['head']['weight'] = init_weights_['head']['weight'][:tokenizer.get_vocab_size(), :]  # type: ignore
# dataset = "english"
# train_str = custom_dataset_str.load(data_path, dataset)
# tokenizer = Tokenizer.from_file(str(model_path / "20B_tokenizer.json"))

n = int(len(train_str) * 0.9)
train_data = train_str[:n]
valid_data = train_str[n:]

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
    'save_interval': 10000,
    'max_iters': 1000000,
    # 'adam': adam_params,
    'lion': lion_params,
    # 'adamw': adam_params,
    'optimizer': 'lion',
}

experimental_params: dict = {
    'eps': 1e-5,
    'n_tokens': tokenizer.get_vocab_size(),
    'n_channels': n_channels,
    'n_blocks': len(init_weights_['blocks']),

    'train_tags': [l.fmt() for l in train_tags] if train_tags is not None else None,

    'batch_size': 4,
    'block_size': 128,
    'train': train_params,
    'model': "rwkv"
}

max_iters = experimental_params['train']['max_iters']
eval_interval = experimental_params['train']['eval_interval']
save_interval = experimental_params['train']['save_interval']
eval_iters = experimental_params['train']['eval_iters']
batch_config_ = BatchConfig(block_size=experimental_params['block_size'],
                            batch_size=experimental_params['batch_size'],
                            tokenizer=tokenizer)

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
train_config_ = TrainConfig(loss_fn=partial(get_lm_loss, rwkv_f),
                            optimiser=optimizer_)
# noinspection PyArgumentList
# cuz it's a NamedTuple
train_state_: TrainState = TrainState(weights=init_weights_,
                                      train_mask=weight_mask,
                                      opt_state=optimizer_.init(init_weights_))

rnn_init_state = np.zeros((experimental_params['n_blocks'], 5, experimental_params['n_channels']))
for i in range(experimental_params['n_blocks']):
    # to jax state[5 * i + 4] = -1e30
    rnn_init_state = rnn_init_state.at[i, 4].set(-1e30)

run = wandb.init(
    project="inside-transformer",
    config=experimental_params,
)
assert isinstance(run, wandb.sdk.wandb_run.Run)

keys_ = next(key_gen).split(max_iters)
for step in range(max_iters):
    batch_ = batch_config_.sample(train_data, keys_[step])
    # batch_ = batch_config_.sample_from_str(train_data, keys_[step])
    if step % eval_interval == 0:
        loss = train_config_.loss_fn(train_state_.weights, batch_)
        print(f"\n===[ step {step} is an eval step ]==========")
        print(f"before step {step}, batch loss {loss}")

    train_state_ = train_config_.train1(train_state_, batch_)
    if step % eval_interval == 0:
        loss = train_config_.loss_fn(train_state_.weights, batch_)
        print(f"after step {step}, batch loss {loss}")
        results = train_config_.estimate_loss(eval_iters, key_gen, train_state_, batch_config_,
                                              {'train': train_data, 'val': valid_data})
        # results = train_config_.estimate_loss_str(eval_iters, key_gen, train_state_, batch_config_,
        #                                       {'train': train_data, 'val': valid_data})
        generate_f = partial(rwkv_rnn, train_state_.weights)
        generated = nlp_utils.rnn_generate(generate_f,
                                           "\n",
                                           tokenizer=batch_config_.tokenizer,
                                           key_gen=key_gen,
                                           init_state=rnn_init_state,
                                           length_per_trial=batch_config_.block_size - 1)

        wandb.log({"train_loss": results['train'],
                   "validation_loss": results['val'],
                   "batch_loss": loss,
                   "n_tokens_trained": step * batch_config_.batch_size * batch_config_.block_size,
                   'generated': wandb.Html(f"{generated}")})
    if step % save_interval == 0:  # and step > 0:
        n_tokens_trained = step * batch_config_.batch_size * batch_config_.block_size
        n_tokens_trained_str = num_short_form(n_tokens_trained)
        wandb.save(save_pytree_(train_state_.weights, run.dir, f"{model_name}_{n_tokens_trained}"), run.dir)

wandb.finish()

# [Done] add trainable weights (via gradient mask)
# [TODO] add trainable weights via weight name mask
# [Done] add checkpointing
# TODO: add weight decay
