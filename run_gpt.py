# %%
from __future__ import annotations

from collections import defaultdict

import jax.numpy as jnp
from optax import softmax_cross_entropy_with_integer_labels

from safetensors import safe_open
from pathlib import Path

from bpe_encoder import get_encoder
from clean_frame import LN, Linear
from gpt import GptMha, GptFfn, GptBlock, GptDecoder, Gpt
from clean_frame_utils import Arr, load_config
from tqdm import tqdm

basic = False
if basic:
    def gpt_loss(gpt: Gpt, inputs: list[int], labels: list[int]) -> Arr:
        logits = go(gpt, (jnp.array(inputs)))
        return softmax_cross_entropy_with_integer_labels(logits, jnp.array(labels))


    def go(c, x: Arr) -> Arr:
        return c.f(c.init_params(), x)


    kkk = go(GptMha.Config(n_channels=9,
                           n_heads=3,
                           T=5).make(), jnp.ones((5, 9)))

    print(kkk.shape)
    gpt_ = Gpt.Config(eps=1e-5,
                      n_channels=9,
                      n_heads=3,
                      T=5,
                      n_blocks=2,
                      n_tokens=10).make()
    zz = go(gpt_, jnp.ones((5,), dtype=jnp.int32))

    print(zz.shape)
    print(gpt_loss(gpt_, [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]))
    www = gpt_.init_params()

gpt_config = Gpt.Config(eps=1e-5,
                        n_channels=768,
                        n_heads=12,
                        T=1024,
                        n_blocks=12,
                        n_tokens=50257,
                        te_name='wte.weight',
                        pe_name='wpe.weight',
                        ln=LN.Config(w_name='weight', b_name='bias', name='ln_f'),
                        name="",
                        decoder=GptDecoder.Config(
                            name='h',
                            blocks=GptBlock.Config(
                                name="",
                                mha=GptMha.Config(
                                    name='attn',
                                    QKV_linear=Linear.Config(name='c_attn', w_name='weight', b_name='bias'),
                                    linear=Linear.Config(name="c_proj", w_name='weight', b_name='bias'),
                                ),
                                ln1=LN.Config(w_name='weight', b_name='bias', name='ln_1'),
                                ffn=GptFfn.Config(
                                    name='mlp',
                                    linear1=Linear.Config(w_name='weight', b_name='bias', name='c_fc'),
                                    linear2=Linear.Config(w_name='weight', b_name='bias', name='c_proj'),
                                ),
                                ln2=LN.Config(w_name='weight', b_name='bias', name='ln_2')
                            )
                        )).fill()

print(gpt_config)
print(gpt_config.weights)
print(gpt_config.parts)

z = defaultdict(int)
ww = load_config(gpt_config, lambda i: z[i])
print(z.keys())

path = Path("/Data/lm_models/gpt2")
with safe_open(path / "model.safetensors", framework="flax", device="cpu") as f:
    weights_tree_ = load_config(gpt_config, f.get_tensor)
    print(weights_tree_.keys())

gpt_ = gpt_config.make()
checked_weights = gpt_config.weights_check(weights_tree_)


# r = gpt_.f(checked_weights, jnp.ones((1024,), dtype=jnp.int32))
# print(r)


def run(inputs):
    return gpt_.f_dyn(checked_weights, jnp.array(inputs))


def generate(inputs, n_tokens_to_generate):
    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = run(inputs)
        next_id = jnp.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate:]  # only return generated ids


encoder = get_encoder("gpt2", "/Data/lm_models/", "vocab.json", "merges.txt")

prompt = "Alan Turing theorized that computers would one day become"
input_ids = encoder.encode(prompt)
output_ids = generate(input_ids, 8)
output_text = encoder.decode(output_ids)
print(output_text)

# TODO:
# [done] implement weight check to parse a weight tree to proper weights
# make shape a dict
# [half] move out init_params
# input/output shapes in config
# [done] fix layernorm issue: clever loading
# make dyn a config instead of separate function
# compare with pico to fix stuff
# change name to save_name