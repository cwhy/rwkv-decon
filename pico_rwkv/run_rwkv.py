from pathlib import Path

import jax.numpy as np
from jax import random
from jax.nn import softmax
from safetensors import safe_open
from tokenizers import Tokenizer

from nlp_utils import rnn_generate
from pico_rwkv.pico_rwkv_parallel import rwkv_net_rnn
from pico_rwkv.pico_rwkv_weights import parse_rwkv_weight
from picojax.random_utils import infinite_safe_keys
from pico_rwkv.pico_rwkv import rwkv_net_w

path = Path("/Data/lm_models/rwkv")
model_name = 'RWKV-4-Pile-430M-20220808-8066'
# jax.config.update('jax_platform_name', 'cpu')


with safe_open(path / f"{model_name}.safetensors", framework="flax", device="cpu") as f:
    w = parse_rwkv_weight(f.keys(), f.get_tensor)

tokenizer = Tokenizer.from_file(str(path / "20B_tokenizer.json"))

n_channels = 1024
ffn_ratio = 4
n_layers = 24

context = ("\nPumas are large, cat-like animals found in America. When reports came into London Zoo that "
           "a wild puma had been spotted forty-five miles south of London, they were not taken seriously."
           " However, as the evidence began to accumulate, experts from the Zoo felt obliged to investigate,"
           " for the descriptions given by people who claimed to have seen the puma were extraordinarily similar."
           "\nThe hunt for the puma began in a small village where a woman picking blackberries saw 'a large cat'"
           " only five yards away from her. It")

# context = "\nPumas are large, "


def sample_rwkv_rnn(token_arr, state):
    return rwkv_net_w(token_arr, state, w)


state = np.zeros((n_layers, 5, n_channels))
for i in range(n_layers):
    # to jax state[5 * i + 4] = -1e30
    state = state.at[i, 4].set(-1e30)


keygen = infinite_safe_keys(0)
rnn_generate(sample_rwkv_rnn, context, state, tokenizer, keygen,
             n_trials=1,
             argmax=False,
             temperature=1.0,
             top_p=0.85,
             length_per_trial=100)

