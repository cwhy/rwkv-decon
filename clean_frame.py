# %%
from __future__ import annotations

import math
from collections import defaultdict
from typing import NamedTuple, TypeVar, Callable, List, TypedDict, Literal, cast, \
    Optional

import jax
import jax.numpy as jnp
from chex import assert_shape
from jax.experimental.maps import xmap
from jax.lax import rsqrt
from jax.nn import softmax
from jax.numpy import mean, var, sqrt, tanh, pi
from optax import softmax_cross_entropy_with_integer_labels

from clean_frame_utils import Arr, load_config, PartsDict, WeightConfig, config_weights_check_, WeightsTree, \
    WeightConfigDict, check_config

C = TypeVar('C')
W = TypeVar('W')


# TODO: unify init_param
# TODO: load real GPT weights

def no_w(d: C) -> tuple[list, C]:
    return [...], d


def batch_ops(f: Callable[[W, Arr], Arr], label: str, add_behind: bool) -> Callable[[W, Arr], Arr]:
    if add_behind:
        extension = [None, label]
    else:
        extension = [label, None]
    return xmap(f, no_w(extension), extension)


def for_all_T(f: Callable[[W, Arr], Arr]) -> Callable[[W, Arr], Arr]:
    return batch_ops(f, 'T', False)


def gelu(x: Arr) -> Arr:
    return 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x ** 3)))


class Linear:
    class Config(NamedTuple):
        n_in: Optional[int] = None
        n_out: Optional[int] = None
        w_name: str = "w"
        b_name: str = "b"
        w_init: Literal["kaiming"] = "kaiming"
        b_init: Literal[0] = 0
        name: str = "linear"

        def make(self) -> Linear:
            check_config(self)
            return Linear(self)

        @property
        def weights(self) -> WeightConfigDict:
            assert self.n_in is not None
            assert self.n_out is not None
            return dict(
                w=WeightConfig(
                    name=self.w_name,
                    shape=(self.n_in, self.n_out),
                    init=self.w_init
                ),
                b=WeightConfig(
                    name=self.b_name,
                    shape=(self.n_out,),
                    init=self.b_init)
            )

        @property
        def parts(self) -> PartsDict:
            return {}

        def weights_check(self, w: WeightsTree) -> Linear.Weights:
            config_weights_check_(self, w)
            return cast(Linear.Weights, w)

    class Weights(TypedDict):
        w: Arr
        b: Arr

    def __init__(self, config: Config):
        self.config = config
        assert config.n_in is not None
        assert config.n_out is not None
        self.n_in = config.n_in
        self.n_out = config.n_out

    def init_params(self):
        return self.Weights(
            w=jnp.zeros((self.n_in, self.n_out)),
            b=jnp.zeros((self.n_out,))
        )

    def f(self, w: Weights, x: Arr) -> Arr:
        assert_shape(x, (self.n_in,))
        result = w['w'].T @ x + w['b']
        assert_shape(result, (self.n_out,))
        return result


class GptMha:
    class Config(NamedTuple):
        n_channels: Optional[int] = None
        n_heads: Optional[int] = None
        T: Optional[int] = None
        linear: Linear.Config = Linear.Config()
        QKV_name: str = "QKV"
        QKV_init: Literal["normal"] = "normal"
        QKV_scale: float = 0.02

        name: str = "mha"

        def fill(self) -> GptMha.Config:
            new = self._replace(linear=self.linear._replace(n_in=self.n_channels, n_out=self.n_channels))
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
            filled = self.fill()
            assert filled.n_channels is not None
            assert filled.n_heads is not None
            assert filled.dim_heads is not None
            return dict(
                QKV=WeightConfig(
                    name=filled.QKV_name,
                    shape=(3, filled.n_heads, filled.dim_heads, filled.n_channels),
                    init=filled.QKV_init,
                    scale=filled.QKV_scale
                ),
            )

        @property
        def parts(self) -> PartsDict:
            filled = self.fill()
            return dict(
                linear=filled.linear
            )

        def weights_check(self, w: WeightsTree) -> GptMha.Weights:
            config_weights_check_(self, w)
            return cast(GptMha.Weights, w)

    class Weights(TypedDict):
        QKV: Arr
        linear: Linear.Weights

    def causal_dot_attention(self, q: Arr, k: Arr, v: Arr) -> Arr:
        assert_shape([q, k, v], (self.dim_heads, self.T))
        result = softmax((q.T @ k) / self.scale + self.mask) @ v.T
        assert_shape(result, (self.T, self.dim_heads))
        return result

    def __init__(self, config: Config):
        assert config.n_channels is not None
        assert config.n_heads is not None
        assert config.T is not None
        assert config.dim_heads is not None
        self.n_channels = config.n_channels
        self.n_heads = config.n_heads
        self.T = config.T
        self.dim_heads = config.dim_heads

        self.linear = config.linear.make()
        self.scale = math.sqrt(self.n_channels)
        self.mask = jnp.tril(jnp.ones((self.T, self.T)))
        self.linearf = for_all_T(self.linear.f)
        assert self.n_channels % self.n_heads == 0, 'n_channels must be divisible by n_heads'

    def init_params(self):
        return GptMha.Weights(
            QKV=jnp.zeros((3, self.n_heads, self.dim_heads, self.n_channels)),
            linear=self.linear.init_params(),
        )

    def f(self, w: GptMha.Weights, x: Arr) -> Arr:
        assert_shape(x, (self.T, self.n_channels))

        q, k, v = w['QKV'] @ x.T
        extension_shape = ['n_head', ...]
        attended = xmap(self.causal_dot_attention, [extension_shape] * 3, extension_shape)(q, k, v)
        assert_shape(attended, (self.n_heads, self.T, self.dim_heads))
        concatenated = jnp.concatenate(attended, -1)
        assert_shape(concatenated, (self.T, self.n_channels))

        result = self.linearf(w['linear'], concatenated)

        assert_shape(result, (self.T, self.n_channels))
        return result


class LN:
    class Config(NamedTuple):
        eps: Optional[float] = None
        shape: Optional[tuple[int, ...]] = None
        norm_dim: Optional[int] = None
        w_name: str = "w"
        b_name: str = "b"
        w_init: Literal[0] = 0
        b_init: Literal[0] = 0
        name: str = "ln"
        norm_dim_name: str = "norm_dim"

        def make(self) -> LN:
            check_config(self)
            return LN(self)

        @property
        def weights(self) -> WeightConfigDict:
            assert self.shape is not None, 'shape must be specified'
            assert self.eps is not None, 'eps must be specified'
            return dict(
                w=WeightConfig(
                    name=self.w_name,
                    shape=self.shape,
                    init=self.w_init
                ),
                b=WeightConfig(
                    name=self.b_name,
                    shape=self.shape,
                    init=self.b_init
                )
            )

        @property
        def parts(self) -> PartsDict:
            return {}

        def weights_check(self, w: WeightsTree) -> LN.Weights:
            config_weights_check_(self, w)
            return cast(LN.Weights, w)

    class Weights(TypedDict):
        w: Arr
        b: Arr

    def __init__(self, config: Config):
        assert config.eps is not None
        self.config = config
        self.eps = config.eps

    def init_params(self) -> LN.Weights:
        return LN.Weights(
            w=jnp.zeros(self.config.weights['w'].shape),
            b=jnp.zeros(self.config.weights['b'].shape)
        )

    # x: self.shape
    def f(self, w: LN.Weights, x: Arr) -> Arr:
        o = x - mean(x, axis=self.config.norm_dim)
        i = w['w'] * rsqrt(var(x, axis=self.config.norm_dim) + self.eps)
        return o * i + w['b']



class GptFfn:
    class Config(NamedTuple):
        n_channels: Optional[int] = None
        T: Optional[int] = None
        linear1: Linear.Config = Linear.Config()
        linear2: Linear.Config = Linear.Config()
        name: str = "ffn"

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
            config_weights_check_(self, w)
            return cast(GptFfn.Weights, w)

    class Weights(TypedDict):
        linear1: Linear.Weights
        linear2: Linear.Weights

    def __init__(self, config: Config):
        self.n_channels = config.n_channels
        self.linear1 = config.linear1.make()
        self.linear2 = config.linear2.make()

    def init_params(self):
        return GptFfn.Weights(
            linear1=self.linear1.init_params(),
            linear2=self.linear2.init_params(),
        )

    def f(self, w: GptFfn.Weights, x: Arr) -> Arr:
        assert_shape(x, (self.n_channels,))
        result = self.linear2.f(w['linear2'], gelu(self.linear1.f(w['linear1'], x)))
        assert_shape(result, (self.n_channels,))
        return result


class GptBlock:
    class Config(NamedTuple):
        eps: Optional[float] = None
        n_channels: Optional[int] = None
        n_heads: Optional[int] = None
        T: Optional[int] = None
        mha: GptMha.Config = GptMha.Config()
        ffn: GptFfn.Config = GptFfn.Config()
        ln1: LN.Config = LN.Config()
        ln2: LN.Config = LN.Config()
        name: str = "gpt_block"

        def fill(self) -> GptBlock.Config:
            new = self._replace(
                mha=self.mha._replace(n_channels=self.n_channels, T=self.T, n_heads=self.n_heads).fill(),
                ffn=self.ffn._replace(n_channels=self.n_channels, T=self.T).fill(),
                ln1=self.ln1._replace(eps=self.eps),
                ln2=self.ln2._replace(eps=self.eps))
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
            config_weights_check_(self, w)
            return cast(GptBlock.Weights, w)

    class Weights(TypedDict):
        mha: GptMha.Weights
        ffn: GptFfn.Weights
        ln1: LN.Weights
        ln2: LN.Weights

    def __init__(self, config: Config):
        self.T = config.T
        self.n_channels = config.n_channels
        self.mha = config.mha.make()
        self.ffn = config.ffn.make()
        self.ln1 = config.ln1.make()
        self.ln2 = config.ln2.make()

        self.ffnf = for_all_T(self.ffn.f)
        self.ln1f = for_all_T(self.ln1.f)
        self.ln2f = for_all_T(self.ln2.f)

    def init_params(self):
        return GptBlock.Weights(
            mha=self.mha.init_params(),
            ffn=self.ffn.init_params(),
            ln1=self.ln1.init_params(),
            ln2=self.ln2.init_params(),
        )

    def f(self, w: GptBlock.Weights, x: Arr) -> Arr:
        assert_shape(x, (self.T, self.n_channels))
        x += self.mha.f(w['mha'], self.ln1f(w['ln1'], x))
        x += self.ffnf(w['ffn'], self.ln2f(w['ln2'], x))

        assert_shape(x, (self.T, self.n_channels))
        return x


class GptDecoder:
    class Config(NamedTuple):
        eps: Optional[float] = None
        n_channels: Optional[int] = None
        n_heads: Optional[int] = None
        T: Optional[int] = None
        n_blocks: Optional[int] = None
        blocks: GptBlock.Config = GptBlock.Config()

        name: str = 'gpt_decoder'

        def fill(self) -> GptDecoder.Config:
            new = self._replace(blocks=self.blocks._replace(eps=self.eps, n_channels=self.n_channels,
                                                            n_heads=self.n_heads, T=self.T).fill())
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
            config_weights_check_(self, w)
            return cast(GptDecoder.Weights, w)

    class Weights(TypedDict):
        blocks: List[GptBlock.Weights]

    def __init__(self, config: Config):
        assert config.n_blocks is not None
        self.T = config.T
        self.n_channels = config.n_channels
        self.blocks = [config.blocks.make() for _ in range(config.n_blocks)]

    def init_params(self):
        return GptDecoder.Weights(
            blocks=[blk.init_params() for blk in self.blocks]
        )

    def f(self, w: GptDecoder.Weights, x: Arr) -> Arr:
        assert_shape(x, (self.T, self.n_channels))
        for blk, blk_w in zip(self.blocks, w['blocks']):
            x = blk.f(blk_w, x)
        assert_shape(x, (self.T, self.n_channels))
        return x


class Gpt:
    class Config(NamedTuple):
        eps: Optional[float] = None
        n_channels: Optional[int] = None
        n_heads: Optional[int] = None
        T: Optional[int] = None
        n_blocks: Optional[int] = None
        n_tokens: Optional[int] = None

        te_name: str = 'te'
        te_init: Literal['normal'] = 'normal'
        te_scale: float = 0.02

        decoder: GptDecoder.Config = GptDecoder.Config()
        ln: LN.Config = LN.Config()

        pe_name: str = 'pe'

        name: str = 'gpt'

        def fill(self) -> Gpt.Config:
            new = self._replace(decoder=self.decoder._replace(eps=self.eps, n_channels=self.n_channels,
                                                              n_heads=self.n_heads, T=self.T,
                                                              n_blocks=self.n_blocks).fill(),
                                ln=self.ln._replace(eps=self.eps))

            check_config(new)
            return new

        def make(self) -> Gpt:
            return Gpt(self.fill())

        @property
        def weights(self) -> WeightConfigDict:
            filled = self.fill()
            assert filled.n_tokens is not None
            assert filled.n_channels is not None
            assert filled.T is not None
            return dict(
                te=WeightConfig(name=filled.te_name,
                                init=filled.te_init,
                                shape=(filled.n_tokens, filled.n_channels),
                                scale=filled.te_scale),

                pe=WeightConfig(name=filled.pe_name,
                                shape=(filled.T, filled.n_channels)),
            )

        @property
        def parts(self) -> PartsDict:
            filled = self.fill()
            assert filled.decoder is not None
            assert filled.ln is not None
            assert filled.te_name is not None
            return dict(
                decoder=filled.decoder,
                ln=filled.ln,
            )

        def weights_check(self, w: WeightsTree) -> Gpt.Weights:
            config_weights_check_(self, w)
            return cast(Gpt.Weights, w)

    class Weights(TypedDict):
        te: Arr
        pe: Arr
        decoder: GptDecoder.Weights
        ln: LN.Weights

    def __init__(self, config: Config):
        assert config.n_blocks is not None
        assert config.n_tokens is not None
        assert config.n_channels is not None
        self.T = config.T
        self.n_channels = config.n_channels
        self.n_tokens = config.n_tokens
        self.eps = config.eps
        self.decoder = config.decoder.make()
        self.ln = config.ln._replace(eps=config.eps).make()

        self.lnf = for_all_T(self.ln.f)

    def init_params(self):
        return Gpt.Weights(
            te=jax.random.normal(jax.random.PRNGKey(0), (self.n_tokens, self.n_channels)),
            pe=jax.random.normal(jax.random.PRNGKey(1), (self.T, self.n_channels)),
            decoder=self.decoder.init_params(),
            ln=self.ln.init_params(),
        )

    def f(self, w: Gpt.Weights, x: Arr) -> Arr:
        assert_shape(x, (self.T,))
        x = w['te'][x, :] + w['pe'][self.T, :]
        assert_shape(x, (self.T, self.n_channels))
        result = self.lnf(w['ln'], self.decoder.f(w['decoder'], x))
        assert_shape(result, (self.T, self.n_channels))
        return result


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

# print(www)


# %%
gpt_config = Gpt.Config(eps=1e-5,
                        n_channels=768,
                        n_heads=12,
                        T=1024,
                        n_blocks=12,
                        n_tokens=10,
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
                                    QKV_name='c_attn.weight',
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

z = defaultdict(int)
ww = load_config(gpt_config, lambda i: z[i])
print(z.keys())

# %%
from safetensors import safe_open
from pathlib import Path

# %%
path = Path("/Data/lm_models/gpt2")
with safe_open(path / "model.safetensors", framework="flax", device="cpu") as f:
    def get_tensor(name):
        if name.endswith("c_attn.weight"):
            return f.get_tensor(name).reshape((3, 12, 64, 768))
        else:
            return f.get_tensor(name)


    weights_tree_ = load_config(gpt_config, get_tensor)
    print(weights_tree_.keys())

# %%
gpt_ = gpt_config.make()
gpt_.f(gpt_config.weights_check(weights_tree_), jnp.ones((1024,), dtype=jnp.int32))

# TODO:
# [done] implement weight check to parse a weight tree to proper weights
# make shape a dict
# fix layernorm issue
