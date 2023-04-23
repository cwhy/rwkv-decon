from pathlib import Path

import jax
import jax.numpy as np
from jax import random
from jax.nn import softmax
from safetensors import safe_open
from tokenizers import Tokenizer

from jax_init_utils import infinite_safe_keys
from pico_rwkv.pico_rwkv import rwkv_net_w
from pico_rwkv.pico_rwkv_parallel import rwkv_net_scan, rwkv_net_rnn

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

for i in w['blocks'].keys():
    w['blocks'][i]['att']['kw'] = w['blocks'][i]['att']['key']['weight']
    w['blocks'][i]['att']['vw'] = w['blocks'][i]['att']['value']['weight']
    w['blocks'][i]['att']['rw'] = w['blocks'][i]['att']['receptance']['weight']
    w['blocks'][i]['ffn']['kw'] = w['blocks'][i]['ffn']['key']['weight']
    w['blocks'][i]['ffn']['vw'] = w['blocks'][i]['ffn']['value']['weight']
    w['blocks'][i]['ffn']['rw'] = w['blocks'][i]['ffn']['receptance']['weight']


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
context = "The quick brown fox jumps over the lazy"

max_len_ = 8
state = np.zeros((n_layers, 5, n_channels))
for i in range(n_layers):
    state = state.at[i, -1].set(-1e30)


parallel = True
if parallel:
    # def pad_tokens(token_objs, max_len):
    #     if len(token_objs.ids) < max_len:
    #         l_token_array = np.array([0] * (max_len - len(token_objs.ids)) + token_objs.ids, dtype=np.int32)
    #     else:
    #         l_token_array = np.array(token_objs.ids, dtype=np.int32)
    #     return l_token_array[-max_len:]

    # token_array = pad_tokens(tokenizer.encode(context), max_len_)
    token_array = np.array(tokenizer.encode(context).ids)

    init_out, state, states0 = rwkv_net_scan(max_len_, token_array, state, **w)
    print(tokenizer.decode(np.argmax(init_out, axis=-1)))

    init_out = init_out[-1, :]
    print(states0[0, :, :5])

    state = np.zeros((n_layers, 5, n_channels))
    for i in range(n_layers):
        state = state.at[i, -1].set(-1e30)

    outs = []
    for token in tokenizer.encode(context).ids:
        init_out, state = rwkv_net_rnn(token, state, **w)
        print(state[0, :, :5])
        raise Exception("stop")
        outs.append(np.argmax(init_out))
    print(tokenizer.decode(outs))
else:
    for token in tokenizer.encode(context).ids:
        init_out, state = rwkv_net_rnn(token, state, **w)

#%%

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
        # token_array = pad_tokens(tokenizer.encode(token), max_len_)
        out, state = rwkv_net_rnn(token, state, **w)
        # print(state[:2, :, :3])
        # out, state = rwkv_net_w(token, state, w)
        # print(state[:2, :, :3])
        # raise Exception("stop")
