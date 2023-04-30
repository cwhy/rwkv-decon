from pathlib import Path

import jax.numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

from nlp_utils import rnn_generate
from pico_rwkv.pico_rwkv_weights import parse_rwkv_weight
from picojax.jax_utils import Arr
from picojax.random_utils import infinite_safe_keys
from pico_rwkv.pico_rwkv_parallel import rwkv_net_parallel, rwkv_net_rnn, rwkv_net_scan

path = Path("/Data/lm_models/rwkv")
model_name = 'RWKV-4-Pile-430M-20220808-8066'
# jax.config.update('jax_platform_name', 'cpu')

with safe_open(path / f"{model_name}.safetensors", framework="flax", device="cpu") as f:
    w = parse_rwkv_weight(f.keys(), f.get_tensor)

tokenizer = Tokenizer.from_file(str(path / "20B_tokenizer.json"))

n_channels = 1024
ffn_ratio = 4
n_layers = 24

key_gen = infinite_safe_keys(0)

# context = """Where must the puma have come from?
#
#     Pumas are large, cat-like animals which are found in America. When reports came into London Zoo that a wild puma had been spotted forty-five miles south of London, they were not taken seriously. However, as the evidence began to accumulate, experts from the Zoo felt obliged to investigate, for the descriptions given by people who claimed to have seen the puma were extraordinarily similar.
#     The hunt for the puma began in a small village where a woman picking blackberries saw 'a large cat' only five yards away from her. It immediately ran away when she saw it, and experts confirmed that a puma will not attack a human being unless it is cornered. The search proved difficult, for the puma was often observed at one place in the morning and at another place twenty miles away in the evening. Wherever it went, it left behind it a trail of dead deer and small animals like rabbits. Paw prints were seen in a number of places and puma fur was found clinging to bushes. Several people complained of "cat-like noises' at night and a businessman on a fishing trip saw the puma up a tree. The experts were now fully convinced that the animal was a puma, but where had it come from? As no pumas had been reported missing from any zoo in the country, this one must have been in the possession of a private collector and somehow managed to escape. The hunt went on for several weeks, but the puma was not caught. It is disturbing to think that a dangerous wild animal is still at large in the quiet countryside.
#
#     In not more than 80 words describe how experts came to the conclusion that the animal seen by many people really was a puma. Do not include anything that is not in the passage.
# Answer these questions in note form to get your points:
# 1  What sort of reports were received by London Zoo?
# 2  Were the reports similar in nature or not?
# 3  Who saw it first?
# 4  Did it stay in one place,or did it move from place to place?
# 5  What did it leave behind it?
# 6  Were paw prints and puma fur found as well or not?
# 7  What was heard at night?
# 8  Was the animal seen up a tree or not?
# 9  Were experts now sure that the animal really was a puma or not?
#
# """


context = ("\nPumas are large, cat-like animals found in America. When reports came into London Zoo that "
           "a wild puma had been spotted forty-five miles south of London, they were not taken seriously."
           " However, as the evidence began to accumulate, experts from the Zoo felt obliged to investigate,"
           " for the descriptions given by people who claimed to have seen the puma were extraordinarily similar."
           "\nThe hunt for the puma began in a small village where a woman picking blackberries saw 'a large cat'"
           " only five yards away from her. It")
# context = "The quick brown fox jumps over the lazy"
# context = "Once upon a "


# mode = "scan"
mode = "rnn"
# mode = "parallel"
if mode == "parallel":
    token_array = np.array(tokenizer.encode(context).ids)
    # init_out = rwkv_net_parallel(len(token_array), token_array, **w)
    # init_out = init_out[-1, :]
    print("token length: ", len(token_array))
    print(context, end="")

    for i in range(100):
        init_out = rwkv_net_parallel(len(token_array), token_array, **w)
        out = np.argmax(init_out[-1, :], axis=-1)
        print(tokenizer.decode([out]), end="", flush=True)
        token_array = np.append(token_array[1:], out)
    raise Exception("Done")


elif mode == "rnn":
    state = np.zeros((n_layers, 5, n_channels))
    for i in range(n_layers):
        state = state.at[i, -1].set(-1e30)
    def sample_rwkv_rnn(token_arr: Arr, state: Arr) -> tuple[Arr, Arr]:
        return rwkv_net_rnn(token_arr, state, **w)
    key_gen = infinite_safe_keys(0)
    _ = rnn_generate(sample_rwkv_rnn, context, state, tokenizer, key_gen,
                     argmax=True,
                     length_per_trial=100, n_trials=3, temperature=1.0, top_p=0.85)

elif mode == "scan":
    max_len_ = 10
    state = np.zeros((n_layers, 5, n_channels))
    for i in range(n_layers):
        state = state.at[i, -1].set(-1e30)


    def process_tokens(token_objs, max_len):
        # l_token_array = np.array_split(full_token_array, len(full_token_array) // max_len)
        # if len(l_token_array[-1]) < max_len:
        #     return l_token_array[:-1], l_token_array[-1]
        # else:
        #     return l_token_array, []
        curr_len = len(token_objs)
        batch = []
        while curr_len > max_len:
            batch.append(np.array(token_objs[:max_len].ids))
            token_objs = token_objs[max_len:]
            curr_len = len(token_objs)
        return batch, np.array(token_objs.ids)


    token_array_batch, left_over = process_tokens(tokenizer.encode(context), max_len_)
    print(token_array_batch)
    print(left_over)
    if len(token_array_batch) > 0:
        for token_array in token_array_batch:
            init_out, state = rwkv_net_scan(max_len_, token_array, state, **w)
    if len(left_over) > 0:
        for token in left_over:
            init_out, state = rwkv_net_rnn(token, state, **w)
    else:
        print("scan_out", tokenizer.decode(np.argmax(init_out)))
        init_out = init_out[-1, :]







