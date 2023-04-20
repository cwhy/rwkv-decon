from __future__ import annotations

import math
from functools import cache
from typing import NamedTuple, Optional, cast, TypedDict, List, Literal, Union, Callable

from chex import assert_shape
from einops import rearrange
from jax import numpy as jnp, vmap, random
from jax.lax import scan
from jax.nn import softmax
from tqdm import tqdm

from clean_frame import Linear, for_all_T, gelu, LN
from clean_frame_utils import check_config, WeightConfigDict, PartsDict, WeightsTree, config_weights_check, Arr, \
    WeightConfig, jit_f
from jax_init_utils import SafeKey


class GptMha:
    class Weights(TypedDict):
        QKV_linear: Linear.Weights
        linear: Linear.Weights

    @jit_f
    def causal_dot_attention(self, q: Arr, k: Arr, v: Arr) -> Arr:
        assert_shape([q, k, v], (self.T, self.dim_heads))
        mask = self.get_mask(q.shape[0])
        result = softmax((q @ k.T) / self.scale + mask) @ v
        assert_shape(result, (self.T, self.dim_heads))
        return result

    @jit_f
    def f(self, w: GptMha.Weights, x: Arr) -> Arr:
        assert_shape(x, (self.T, self.n_channels))

        q, k, v = rearrange(self.QKV_linearf(w['QKV_linear'], x),
                            'T (qkv n_heads dim_heads) -> qkv n_heads T dim_heads',
                            qkv=3,
                            n_heads=self.n_heads,
                            dim_heads=self.dim_heads)
        # extension_shape = ['n_head', ...]
        # attended = xmap(self.causal_dot_attention, [extension_shape] * 3, extension_shape)(q, k, v)
        attended = vmap(self.causal_dot_attention, (0, 0, 0), 0)(q, k, v)
        assert_shape(attended, (self.n_heads, self.T, self.dim_heads))
        concatenated = jnp.concatenate(attended, -1)
        assert_shape(concatenated, (self.T, self.n_channels))

        result = self.linearf(w['linear'], concatenated)

        assert_shape(result, (self.T, self.n_channels))
        return result

    def f_debug(self, w: GptMha.Weights, x: Arr) -> dict[str, Arr]:
        assert_shape(x, (self.T, self.n_channels))

        qs, ks, vs = rearrange(self.QKV_linearf(w['QKV_linear'], x),
                               'T (qkv n_heads dim_heads) -> qkv n_heads T dim_heads',
                               qkv=3,
                               n_heads=self.n_heads,
                               dim_heads=self.dim_heads)
        mask = self.get_mask(qs.shape[1])
        attended_list = []
        attn_maps = []
        attn_maps_raw = []
        for q, k, v in zip(qs, ks, vs):
            attn_map_raw = q @ k.T
            attn_maps_raw.append(attn_map_raw)
            attn_map = softmax(attn_map_raw / self.scale + mask)
            attn_maps.append(attn_map)
            attended_list.append(attn_map @ v)
        attended = jnp.stack(attended_list)
        assert_shape(attended, (self.n_heads, self.T, self.dim_heads))
        concatenated = jnp.concatenate(attended, -1)
        assert_shape(concatenated, (self.T, self.n_channels))

        result = self.linearf(w['linear'], concatenated)

        assert_shape(result, (self.T, self.n_channels))
        return dict(
            attn=jnp.stack(attn_maps),
            attn_raw=jnp.stack(attn_maps_raw),
            x_after_mha=result)

    class Config(NamedTuple):
        n_channels: Optional[int] = None
        n_heads: Optional[int] = None
        n_seq: Optional[Union[int, Literal['dynamic']]] = None
        inf_mask: float = -1e10
        linear: Linear.Config = Linear.Config()
        QKV_linear: Linear.Config = Linear.Config()

        save_name: str = "mha"

        @property
        @cache
        def T(self) -> Optional[int]:
            if self.n_seq == 'dynamic':
                return None
            else:
                return self.n_seq

        def fill(self) -> GptMha.Config:
            assert self.n_channels is not None, 'n_channels must be set'
            new = self._replace(linear=self.linear._replace(n_in=self.n_channels, n_out=self.n_channels),
                                QKV_linear=self.QKV_linear._replace(n_in=self.n_channels, n_out=3 * self.n_channels))
            check_config(new)
            return new

        def make(self) -> GptMha:
            return GptMha(self.fill())

        @property
        def dim_heads(self) -> int:
            assert self.n_channels is not None
            assert self.n_heads is not None
            return self.n_channels // self.n_heads

        @property
        def weights(self) -> WeightConfigDict:
            return {}

        @property
        def parts(self) -> PartsDict:
            filled = self.fill()
            return dict(
                linear=filled.linear,
                QKV_linear=filled.QKV_linear
            )

        def weights_check(self, w: WeightsTree) -> GptMha.Weights:
            return cast(GptMha.Weights, config_weights_check(self, w))

    def __init__(self, config: Config):
        assert config.n_channels is not None
        assert config.n_heads is not None
        assert config.dim_heads is not None
        self.n_channels = config.n_channels
        self.n_heads = config.n_heads
        self.T = config.T
        self.dim_heads = config.dim_heads
        self.config = config

        self.linear = config.linear.make()
        self.QKV_linear = config.QKV_linear.make()
        self.scale = math.sqrt(self.dim_heads)
        self.linearf = for_all_T(self.linear.f)
        self.QKV_linearf = for_all_T(self.QKV_linear.f)
        assert self.n_channels % self.n_heads == 0, 'n_channels must be divisible by n_heads'

    def get_mask(self, t: int) -> Arr:
        return (1 - jnp.tri(t)) * self.config.inf_mask


class GptFfn:
    class Weights(TypedDict):
        linear1: Linear.Weights
        linear2: Linear.Weights

    @jit_f
    def f(self, w: GptFfn.Weights, x: Arr) -> Arr:
        assert_shape(x, (self.n_channels,))
        result = self.linear2.f(w['linear2'], gelu(self.linear1.f(w['linear1'], x)))
        assert_shape(result, (self.n_channels,))
        return result

    class Config(NamedTuple):
        n_channels: Optional[int] = None
        n_seq: Optional[Union[int, Literal['dynamic']]] = None
        linear1: Linear.Config = Linear.Config()
        linear2: Linear.Config = Linear.Config()
        save_name: str = "ffn"

        @property
        @cache
        def T(self) -> Optional[int]:
            if self.n_seq == 'dynamic':
                return None
            else:
                return self.n_seq

        def fill(self) -> GptFfn.Config:
            assert self.n_channels is not None
            new = self._replace(
                linear1=self.linear1._replace(n_in=self.n_channels, n_out=self.n_channels * 4),
                linear2=self.linear2._replace(n_in=self.n_channels * 4, n_out=self.n_channels))
            check_config(new)
            return new

        def make(self) -> GptFfn:
            return GptFfn(self.fill())

        @property
        def weights(self) -> WeightConfigDict:
            return {}

        @property
        def parts(self) -> PartsDict:
            filled = self.fill()
            return dict(
                linear1=filled.linear1,
                linear2=filled.linear2
            )

        def weights_check(self, w: WeightsTree) -> GptFfn.Weights:
            return cast(GptFfn.Weights, config_weights_check(self, w))

    def __init__(self, config: Config):
        self.n_channels = config.n_channels
        self.linear1 = config.linear1.make()
        self.linear2 = config.linear2.make()


class GptBlock:
    class Weights(TypedDict):
        mha: GptMha.Weights
        ffn: GptFfn.Weights
        ln1: LN.Weights
        ln2: LN.Weights

    @jit_f
    def f(self, w: GptBlock.Weights, x: Arr) -> Arr:
        assert_shape(x, (self.T, self.n_channels))
        x += self.mha.f(w['mha'], self.ln1f(w['ln1'], x))
        x += self.ffnf(w['ffn'], self.ln2f(w['ln2'], x))
        assert_shape(x, (self.T, self.n_channels))
        return x

    def f_debug(self, w: GptBlock.Weights, x: Arr) -> dict[str, Arr]:
        return_dict = {}
        x0 = x
        return_dict['x0'] = x
        x = self.ln1f(w['ln1'], x)
        return_dict['x_before_mha'] = x
        attn_result = self.mha.f_debug(w['mha'], x)
        return_dict.update(attn_result)
        x = attn_result['x_after_mha']
        x = x0 + x
        x0 = x
        x = self.ln2f(w['ln2'], x)
        return_dict['x_before_ffn'] = x
        x = self.ffnf(w['ffn'], x)
        return_dict['x_after_ffn'] = x
        x = x0 + x
        return_dict['x'] = x
        return return_dict

    class Config(NamedTuple):
        eps: Optional[float] = None
        n_channels: Optional[int] = None
        n_heads: Optional[int] = None
        n_seq: Optional[Union[int, Literal['dynamic']]] = None
        mha: GptMha.Config = GptMha.Config()
        ffn: GptFfn.Config = GptFfn.Config()
        ln1: LN.Config = LN.Config()
        ln2: LN.Config = LN.Config()
        save_name: str = "gpt_block"

        @property
        def T(self) -> Optional[int]:
            if self.n_seq == 'dynamic':
                return None
            else:
                return self.n_seq

        @property
        def x_shape(self) -> tuple[Optional[int], ...]:
            assert self.n_channels is not None
            return self.T, self.n_channels

        def fill(self) -> GptBlock.Config:
            new = self._replace(
                mha=self.mha._replace(n_channels=self.n_channels, n_seq=self.n_seq, n_heads=self.n_heads).fill(),
                ffn=self.ffn._replace(n_channels=self.n_channels, n_seq=self.n_seq).fill(),
                ln1=self.ln1._replace(eps=self.eps, norm_dims=(0,), x_shape=self.x_shape),
                ln2=self.ln2._replace(eps=self.eps, norm_dims=(0,), x_shape=self.x_shape))
            check_config(new)
            return new

        def make(self) -> GptBlock:
            return GptBlock(self.fill())

        @property
        def weights(self) -> WeightConfigDict:
            return {}

        @property
        def parts(self) -> PartsDict:
            filled = self.fill()
            return dict(
                mha=filled.mha,
                ffn=filled.ffn,
                ln1=filled.ln1,
                ln2=filled.ln2,
            )

        def weights_check(self, w: WeightsTree) -> GptBlock.Weights:
            return cast(GptBlock.Weights, config_weights_check(self, w))

    def __init__(self, config: Config):
        self.T = config.T
        self.n_channels = config.n_channels
        self.mha = config.mha.make()
        self.ffn = config.ffn.make()
        self.ln1 = config.ln1.make()
        self.ln2 = config.ln2.make()

        self.ffnf = for_all_T(self.ffn.f)
        self.ln1f = self.ln1.f
        self.ln2f = self.ln2.f


class GptDecoder:
    class Weights(TypedDict):
        blocks: List[GptBlock.Weights]

    @jit_f
    def f(self, w: GptDecoder.Weights, x: Arr) -> Arr:
        assert_shape(x, (self.T, self.n_channels))
        for blk, blk_w in zip(self.blocks, w['blocks']):
            x = blk.f(blk_w, x)
        assert_shape(x, (self.T, self.n_channels))
        return x

    def f_debug(self, w: GptDecoder.Weights, x: Arr) -> tuple[Arr, dict[str, Arr]]:
        assert_shape(x, (self.T, self.n_channels))
        view_vec = []
        for blk, blk_w in zip(self.blocks, w['blocks']):
            return_dict = blk.f_debug(blk_w, x)
            view_vec.append(return_dict)
            x = return_dict['x']
        assert_shape(x, (self.T, self.n_channels))
        vecs = {key: jnp.stack([dict_item[key] for dict_item in view_vec]) for key in view_vec[0].keys()}
        return x, vecs

    class Config(NamedTuple):
        eps: Optional[float] = None
        n_channels: Optional[int] = None
        n_heads: Optional[int] = None
        n_seq: Optional[Union[int, Literal['dynamic']]] = None
        n_blocks: Optional[int] = None
        blocks: GptBlock.Config = GptBlock.Config()

        save_name: str = 'gpt_decoder'

        @property
        @cache
        def T(self) -> Optional[int]:
            if self.n_seq == 'dynamic':
                return None
            else:
                return self.n_seq

        def fill(self) -> GptDecoder.Config:
            new = self._replace(blocks=self.blocks._replace(eps=self.eps, n_channels=self.n_channels,
                                                            n_heads=self.n_heads, n_seq=self.n_seq).fill())
            check_config(new)
            return new

        def make(self) -> GptDecoder:
            return GptDecoder(self.fill())

        @property
        def weights(self) -> WeightConfigDict:
            return {}

        @property
        def parts(self) -> PartsDict:
            filled = self.fill()
            assert filled.blocks is not None
            assert filled.n_blocks is not None
            return dict(
                blocks=[filled.blocks] * filled.n_blocks
            )

        def weights_check(self, w: WeightsTree) -> GptDecoder.Weights:
            return cast(GptDecoder.Weights, config_weights_check(self, w))

    def __init__(self, config: Config):
        assert config.n_blocks is not None
        self.T = config.T
        self.n_channels = config.n_channels
        self.blocks = [config.blocks.make() for _ in range(config.n_blocks)]


class Gpt:
    class Weights(TypedDict):
        token_embedding: Arr
        positional_encoding: Arr
        decoder: GptDecoder.Weights
        ln: LN.Weights

    @jit_f
    def f(self, w: Gpt.Weights, x: Arr) -> Arr:
        assert_shape(x, (self.T,))
        # remove x.shape[0] for faster but static seq len
        if self.T is None:
            x = w['token_embedding'][x, :] + w['positional_encoding'][:x.shape[0], :]
        else:
            x = w['token_embedding'][x, :] + w['positional_encoding'][:self.T, :]
        assert_shape(x, (self.T, self.n_channels))
        result = self.ln.f(w['ln'], self.decoder.f(w['decoder'], x))
        assert_shape(result, (self.T, self.n_channels))
        return result @ w['token_embedding'].T

    def f_debug(self, w: Gpt.Weights, x: Arr) -> tuple[Arr, dict[str, Arr]]:
        assert_shape(x, (self.T,))
        x = w['token_embedding'][x, :] + w['positional_encoding'][:x.shape[0], :]
        assert_shape(x, (self.T, self.n_channels))
        decoder_out, save_vecs = self.decoder.f_debug(w['decoder'], x)
        result = self.ln.f(w['ln'], decoder_out)
        assert_shape(result, (self.T, self.n_channels))
        return result @ w['token_embedding'].T, save_vecs

    class Config(NamedTuple):
        eps: Optional[float] = None
        n_channels: Optional[int] = None
        n_heads: Optional[int] = None
        n_seq: Optional[Union[int, Literal['dynamic']]] = None
        n_blocks: Optional[int] = None
        n_tokens: Optional[int] = None
        max_seq_len: Optional[int] = None

        token_embedding_save_name: str = 'te'
        token_embedding_init: Literal['normal'] = 'normal'
        token_embedding_scale: float = 0.02

        decoder: GptDecoder.Config = GptDecoder.Config()
        ln: LN.Config = LN.Config()

        positional_embedding_save_name: str = 'pe'

        save_name: str = 'gpt'

        @property
        def T(self) -> Optional[int]:
            if self.n_seq == 'dynamic':
                return None
            else:
                return self.n_seq

        def fill(self) -> Gpt.Config:
            assert self.n_channels is not None, 'n_channels must be specified'
            new = self._replace(decoder=self.decoder._replace(eps=self.eps, n_channels=self.n_channels,
                                                              n_heads=self.n_heads, n_seq=self.n_seq,
                                                              n_blocks=self.n_blocks).fill(),
                                ln=self.ln._replace(eps=self.eps, norm_dims=(0,), x_shape=(self.T, self.n_channels)))

            check_config(new)
            return new

        def make(self) -> Gpt:
            return Gpt(self.fill())

        @property
        def weights(self) -> WeightConfigDict:
            filled = self.fill()
            assert filled.max_seq_len is not None
            assert filled.n_tokens is not None
            assert filled.n_channels is not None
            return dict(
                token_embedding=WeightConfig(save_name=filled.token_embedding_save_name,
                                             init=filled.token_embedding_init,
                                             shape=(filled.n_tokens, filled.n_channels),
                                             scale=filled.token_embedding_scale),

                positional_encoding=WeightConfig(save_name=filled.positional_embedding_save_name,
                                                 shape=(filled.max_seq_len, filled.n_channels)),
            )

        @property
        def parts(self) -> PartsDict:
            filled = self.fill()
            assert filled.decoder is not None
            assert filled.ln is not None
            assert filled.token_embedding_save_name is not None
            return dict(
                decoder=filled.decoder,
                ln=filled.ln,
            )

        def weights_check(self, w: WeightsTree) -> Gpt.Weights:
            return cast(Gpt.Weights, config_weights_check(self, w))

    def __init__(self, config: Config):
        assert config.n_blocks is not None
        assert config.n_tokens is not None
        assert config.n_channels is not None
        self.config = config
        self.T = config.T
        self.n_channels = config.n_channels
        self.n_tokens = config.n_tokens
        self.eps = config.eps
        self.decoder = config.decoder.make()
        self.ln = config.ln.make()


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


# from https://github.com/cgarciae/nanoGPT-jax/blob/master/model.py
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


def get_positional_encoding(max_len: int, d_model: int):
    pe = jnp.zeros((max_len, d_model))
    position = jnp.expand_dims(jnp.arange(0, max_len), 1)
    div_term = jnp.exp(
        jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model)
    )
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe


