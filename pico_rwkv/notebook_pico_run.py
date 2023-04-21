import types

import jax
import jax.numpy as np
from pathlib import Path

from tokenizers import Tokenizer
from jax import random
from jax.nn import softmax
from safetensors import safe_open

from jax_init_utils import infinite_safe_keys
from pico_rwkv.pico_rwkv import rwkv_net_w

path = Path("/Data/lm_models/rwkv")
model_name = 'RWKV-4-Pile-430M-20220808-8066'
jax.config.update('jax_platform_name', 'cpu')

w = types.SimpleNamespace()  # set self.w from w
with safe_open(path / f"{model_name}.safetensors", framework="flax", device="cpu") as f:
    w.blocks = {}
    for k in f.keys():  # example: "blocks.0.att.time_first" => self.w.blocks[0].att.time_first
        parts = k.split('.')
        last = parts.pop()
        current_ = w
        for p in parts:
            if p.isdigit():
                p = int(p)
                assert isinstance(current_, dict)
                if p not in current_:
                    current_[p] = types.SimpleNamespace()
                current_ = current_[p]
            else:
                if not hasattr(current_, p):
                    setattr(current_, p, types.SimpleNamespace())
                current_ = getattr(current_, p)
        setattr(current_, last, f.get_tensor(k))

# %%

tokenizer = Tokenizer.from_file(str(path / "20B_tokenizer.json"))

n_channels = 1024
ffn_ratio = 4
n_layers = 10


key_gen = infinite_safe_keys(0)


def sample_logits(out, key, temperature=1.0, top_p=0.8):
    probs = softmax(out, axis=-1)
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs = probs.at[probs < cutoff].set(0)
    if temperature != 1.0:
        probs = np.power(probs, 1.0 / temperature)
    probs = probs / np.sum(probs)

    out = random.categorical(key.get(), probs, axis=-1)
    return out




context = "\nPumas are large, cat-like animals"

tokens = tokenizer.encode(context)

state = np.zeros((n_layers * 5, n_channels))
for i in range(n_layers):
    # to jax state[5 * i + 4] = -1e30
    state = state.at[5 * i + 4].set(-1e30)

for token in tokens.ids:
    init_out, init_state = rwkv_net_w(token, state, w)

NUM_TRIALS = 3
LENGTH_PER_TRIAL = 100
TEMPERATURE = 1.0
TOP_P = 0.85

for TRIAL in range(NUM_TRIALS):
    print(f'\n\n--[ Trial {TRIAL} ]-----------------', context, end="")
    all_tokens = []
    out_last = 0
    out, state = init_out, init_state
    for i in range(LENGTH_PER_TRIAL):
        token = np.argmax(out)
        # token = sample_logits(out, next(key_gen), TEMPERATURE, TOP_P)
        all_tokens.append(token)
        tmp = tokenizer.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:  # only print when we have a valid utf-8 string
            print(tmp, end="", flush=True)
            out_last = i + 1
        out, state = rwkv_net_w(token, state, w)
