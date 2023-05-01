from __future__ import annotations

from typing import Callable, Iterator, Protocol

from jax import numpy as jnp, random, numpy as np
from jax._src.lax.control_flow import scan
from jax._src.nn.functions import softmax
from tqdm import tqdm

from picojax.jax_utils import Arr
from picojax.random_utils import SafeKey


class Tokens(Protocol):
    ids: list[int]


class Tokenizer(Protocol):
    def encode(self, text: str) -> Tokens:
        ...

    def decode(self, ids: list[int]) -> str:
        ...

    def get_vocab_size(self) -> int:
        ...


def generate(get_logits: Callable[[Arr], Arr], inputs: list[int], n_tokens_to_generate: int, max_len: int):
    input_window = inputs
    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = get_logits(jnp.array(input_window))
        next_id = jnp.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input
        input_window = inputs[-max_len:]  # update input window

    return inputs[len(inputs) - n_tokens_to_generate:]  # only return generated ids


def generate_static(get_logits: Callable[[Arr], Arr], inputs: list[int], n_tokens_to_generate: int, max_len: int):
    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        if len(inputs) >= max_len:
            input_window = inputs[-max_len:]  # update input window
        else:
            input_window = inputs + [0] * (max_len - len(inputs))
        output_index = len(inputs) - 1
        logits = get_logits(jnp.array(input_window))
        next_id = jnp.argmax(logits[output_index])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate:]  # only return generated ids


# not working yet from https://github.com/cgarciae/nanoGPT-jax/blob/master/model.py
def generate_static_inplace(get_logits: Callable[[Arr], Arr],
                            key: SafeKey,
                            inputs: list[int],
                            n_tokens_to_generate: int,
                            max_len: int,
                            temperature=1.0,
                            top_k=None):
    input_len = len(inputs)
    input_tokens = jnp.array(inputs)
    padding = jnp.zeros(n_tokens_to_generate, dtype=jnp.int32)
    tokens = jnp.concatenate([input_tokens, padding], axis=-1)
    indexes = jnp.arange(input_len, input_len + n_tokens_to_generate)

    # tokens index -> tokens None
    def scan_f(tokens, i):
        # l: x y
        # t: a b - -
        # i: 0 1 2 3
        step_key = random.fold_in(key.get(), i)
        # if the sequence context is growing too long we must crop it at block_size
        # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        # forward the model to get the logits for the index in the sequence
        logits = get_logits(tokens)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[i - 1] / temperature
        # optionally crop the logits to only the top k options
        # sample from the distribution
        if top_k is not None:
            top_logits, top_tokens = top_k(logits, min(top_k, logits.shape[-1]))
            token_idx = random.categorical(step_key, top_logits, axis=-1)
            next_token = jnp.take_along_axis(top_tokens, token_idx[:, None], axis=-1).squeeze(-1)
        else:
            next_token = random.categorical(step_key, logits, axis=-1)
            # logits = jnp.where(logits < v[:, -1:], float('-inf'), logits)
        # append sampled index to the running sequence and continue
        tokens = tokens.at[i].set(next_token)

        return tokens, None

    tokens, _ = scan(scan_f, tokens, indexes)

    return tokens.tolist()


def rnn_generate(get_logits_rnn: Callable[[Arr, Arr], tuple[Arr, Arr]],
                 context: str,
                 init_state: Arr,
                 tokenizer: Tokenizer,
                 key_gen: Iterator[SafeKey],
                 argmax: bool = False,
                 length_per_trial: int = 100, n_trials: int = 1, temperature: float = 1.0, top_p: float = 0.85) -> str:
    init_state = init_state.copy()
    for token in tokenizer.encode(context).ids:
        init_out, init_state = get_logits_rnn(token, init_state)
        # print(init_state[1, :, 1])

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

    out_str = ""
    for t in range(n_trials):
        to_print = f'\n\n--[ Trial {t} ]-----------------\n{context}'
        print(to_print, end="")
        out_str += to_print
        all_tokens = []
        out_last = 0
        out, state_ = init_out, init_state.copy()
        for i in range(length_per_trial):
            if argmax:
                token = np.argmax(out)
            else:
                token = sample_logits(out, next(key_gen), temperature, top_p)
            all_tokens.append(token)
            tmp = tokenizer.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp:  # only print when we have a valid utf-8 string
                print(tmp, end="", flush=True)
                out_str += tmp
                out_last = i + 1
            out, state_ = get_logits_rnn(token, state_)
    print(flush=True)
    return out_str
