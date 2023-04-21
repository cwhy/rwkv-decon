# %%
from __future__ import annotations

from collections import defaultdict

import jax.numpy as jnp
from jax import jit

from safetensors import safe_open
from pathlib import Path

from safetensors.flax import save_file

from bpe_encoder import get_encoder
from clean_frame import LN, Linear
from gpt import GptMha, GptFfn, GptBlock, GptDecoder, Gpt, generate
from clean_frame_utils import Arr, load_config

gpt_config = Gpt.Config(eps=1e-5,
                        n_channels=768,
                        n_heads=12,
                        n_seq='dynamic',
                        max_seq_len=1024,
                        n_blocks=12,
                        n_tokens=50257,
                        token_embedding_save_name='wte.weight',
                        positional_embedding_save_name='wpe.weight',
                        ln=LN.Config(w_save_name='weight', b_save_name='bias', save_name='ln_f'),
                        save_name="",
                        decoder=GptDecoder.Config(
                            save_name='h',
                            blocks=GptBlock.Config(
                                save_name="",
                                mha=GptMha.Config(
                                    save_name='attn',
                                    QKV_linear=Linear.Config(save_name='c_attn', w_save_name='weight', b_save_name='bias'),
                                    linear=Linear.Config(save_name="c_proj", w_save_name='weight', b_save_name='bias'),
                                ),
                                ln1=LN.Config(w_save_name='weight', b_save_name='bias', save_name='ln_1'),
                                ffn=GptFfn.Config(
                                    save_name='mlp',
                                    linear1=Linear.Config(w_save_name='weight', b_save_name='bias', save_name='c_fc'),
                                    linear2=Linear.Config(w_save_name='weight', b_save_name='bias', save_name='c_proj'),
                                ),
                                ln2=LN.Config(w_save_name='weight', b_save_name='bias', save_name='ln_2')
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
    return gpt_.f(checked_weights, jnp.array(inputs))


def debug(inputs) -> dict[str, Arr]:
    logits, to_save = jit(gpt_.f_debug)(checked_weights, jnp.array(inputs))
    out = encoder.decode([int(jnp.argmax(logits[-1]))])
    print(out)
    save_dir = "../saves"
    return save_file(to_save, f'{save_dir}/view_vec2_dict_jit')


encoder = get_encoder("gpt2", "/Data/lm_models/", "vocab.json", "merges.txt")

# prompt = "Alan Turing theorized that computers would one day become"
prompt = "Time flies like an arrow, fruit flies like a banana. Time flies like an"
# prompt = "Time flies like an arrow; fruit flies like a banana."
# prompt = "The number of pixels used to render an image is set by the Axes size and the dpi of the figure. This can lead to aliasing artifacts when the image is resampled because the displayed image size will usually not match the size of X (see Image antialiasing)."
input_ids = encoder.encode(prompt)
print([encoder.decoder[t] for t in input_ids])

debug(input_ids)

# output_ids = generate(run, input_ids, 8)
# output_text = encoder.decode(output_ids)
# print(output_text)

# TODO:
# [done] implement weight check to parse a weight tree to proper weights
# make shape a dict
# [done] move out init_params
# input/output shapes in config
# [done] fix layernorm issue: clever loading
# [done] make dyn a config instead of separate function
# [done] compare with pico to fix stuff
# change name to save_name
# [half] investigate ffn residual (make good visualizations)
# [done] complete the debug_chain
# make view panel
# [half] better folder structure
