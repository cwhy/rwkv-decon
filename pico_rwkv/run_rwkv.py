from pathlib import Path

import jax.numpy as np
from jax import random
from jax.nn import softmax
from safetensors import safe_open
from tokenizers import Tokenizer

from jax_init_utils import infinite_safe_keys
from pico_rwkv.pico_rwkv import rwkv_net_w

path = Path("/Data/lm_models/rwkv")
model_name = 'RWKV-4-Pile-430M-20220808-8066'
# jax.config.update('jax_platform_name', 'cpu')

w = {}
with safe_open(path / f"{model_name}.safetensors", framework="flax", device="cpu") as f:
    for k in f.keys():
        parts = k.split('.')
        last = parts.pop()
        current_ = w
        for p in parts:
            if p.isdigit():
                p = int(p)
            assert isinstance(current_, dict)
            if p not in current_:
                current_[p] = {}
            current_ = current_[p]
        current_[last] = f.get_tensor(k)

# %%

tokenizer = Tokenizer.from_file(str(path / "20B_tokenizer.json"))

n_channels = 1024
ffn_ratio = 4
n_layers = 24

key_gen = infinite_safe_keys(0)


def sample_logits(logits, key, temperature=1.0, top_p=0.8):
    probs = softmax(logits, axis=-1)
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs = np.where(probs < cutoff, 0, probs)
    if temperature != 1.0:
        probs = np.power(probs, 1.0 / temperature)
    probs = probs / np.sum(probs)

    out = random.choice(key.get(), a=len(probs), p=probs)
    return out


context = ("\nPumas are large, cat-like animals found in America. When reports came into London Zoo that "
           "a wild puma had been spotted forty-five miles south of London, they were not taken seriously."
           " However, as the evidence began to accumulate, experts from the Zoo felt obliged to investigate,"
           " for the descriptions given by people who claimed to have seen the puma were extraordinarily similar."
           "\nThe hunt for the puma began in a small village where a woman picking blackberries saw 'a large cat'"
           " only five yards away from her. It")

tokens = tokenizer.encode(context)

state = np.zeros((n_layers, 5, n_channels))
for i in range(n_layers):
    # to jax state[5 * i + 4] = -1e30
    state = state.at[i, 4].set(-1e30)

for token in tokens.ids:
    init_out, state = rwkv_net_w(token, state, w)

NUM_TRIALS = 3
LENGTH_PER_TRIAL = 100
TEMPERATURE = 1.0
TOP_P = 0.85

for TRIAL in range(NUM_TRIALS):
    print(f'\n\n--[ Trial {TRIAL} ]-----------------', context, end="")
    all_tokens = []
    out_last = 0
    out, state = init_out, state.copy()
    for i in range(LENGTH_PER_TRIAL):
        # token = np.argmax(out)
        token = sample_logits(out, next(key_gen), TEMPERATURE, TOP_P)
        all_tokens.append(token)
        tmp = tokenizer.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:  # only print when we have a valid utf-8 string
            print(tmp, end="", flush=True)
            out_last = i + 1
        out, state = rwkv_net_w(token, state, w)