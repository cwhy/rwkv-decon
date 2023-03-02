# %%
from __future__ import annotations

import math
from typing import NamedTuple, TypeVar, Callable, List, TypedDict

import jax
import jax.numpy as jnp
from chex import assert_shape
from jax.experimental.maps import xmap
from jax.lax import rsqrt
from jax.nn import softmax
from jax.numpy import mean, var, sqrt, tanh, pi
from optax import softmax_cross_entropy_with_integer_labels

Arr = jax.Array
C = TypeVar('C')
W = TypeVar('W')


# TODO: unify init_param
# TODO: load real GPT weights

def no_w(d: C) -> tuple[list, C]:
    return [...], d


def for_all_T(f: Callable[[W, Arr], Arr]) -> Callable[[W, Arr], Arr]:
    add_behind = False
    if add_behind:
        extension = [None, 'T']
    else:
        extension = ['T', None]
    return xmap(f, no_w(extension), extension)


def gelu(x: Arr) -> Arr:
    return 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x ** 3)))


class Linear:
    class Config(TypedDict):
        n_in: int
        n_out: int

    class Weights(TypedDict):
        w: Arr
        b: Arr

    def __init__(self, config: Config):
        self.n_in = config['n_in']
        self.n_out = config['n_out']

    def init_params(self):
        return self.Weights(
            w=jnp.zeros((self.n_out, self.n_in)),
            b=jnp.zeros((self.n_out,))
        )

    def f(self, w: Linear.Weights, x: Arr) -> Arr:
        assert_shape(x, (self.n_in,))
        result = w['w'] @ x + w['b']
        assert_shape(result, (self.n_out,))
        return result


class GptMha:
    class Config(TypedDict):
        n_channels: int
        n_heads: int
        T: int

    class Weights(TypedDict):
        QKV: Arr
        linear: Linear.Weights

    def causal_dot_attention(self, q: Arr, k: Arr, v: Arr) -> Arr:
        assert_shape([q, k, v], (self.dim_heads, self.T))
        result = softmax((q.T @ k) / self.scale + self.mask) @ v.T
        assert_shape(result, (self.T, self.dim_heads))
        return result

    def __init__(self, config: Config):
        self.n_channels = config['n_channels']
        self.n_heads = config['n_heads']
        self.T = config['T']
        self.linear = Linear({'n_in': self.n_channels, 'n_out': self.n_channels})
        self.scale = math.sqrt(self.n_channels)
        self.mask = jnp.tril(jnp.ones((self.T, self.T)))
        self.linearf = for_all_T(self.linear.f)
        assert self.n_channels % self.n_heads == 0, 'n_channels must be divisible by n_heads'
        self.dim_heads: int = self.n_channels // self.n_heads

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
    class Config(TypedDict):
        eps: float

    class Weights(TypedDict):
        w: Arr
        b: Arr

    def __init__(self, config: Config):
        self.eps = config['eps']

    @staticmethod
    def init_params() -> LN.Weights:
        return LN.Weights(
            w=jnp.zeros((1,)),
            b=jnp.zeros((1,))
        )

    # x: (to_normalize,)
    def f(self, w: LN.Weights, x: Arr) -> Arr:
        o = x - mean(x)
        i = w['w'] * rsqrt(var(x) + self.eps)
        return o * i + w['b']


class GptFfn:
    class Config(TypedDict):
        n_channels: int
        T: int

    class Weights(TypedDict):
        linear1: Linear.Weights
        linear2: Linear.Weights

    def __init__(self, config: Config):
        self.n_channels = config['n_channels']
        self.linear1 = Linear({'n_in': self.n_channels, 'n_out': self.n_channels})
        self.linear2 = Linear({'n_in': self.n_channels, 'n_out': self.n_channels})

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
    class Config(TypedDict):
        eps: float
        n_channels: int
        n_heads: int
        T: int

    class Weights(TypedDict):
        mha: GptMha.Weights
        ffn: GptFfn.Weights
        ln1: LN.Weights
        ln2: LN.Weights

    def __init__(self, config: GptBlock.Config):
        self.T = config["T"]
        self.n_channels = config["n_channels"]
        self.mha = GptMha({"n_channels": config["n_channels"], "n_heads": config["n_heads"], "T": config["T"]})
        self.ffn = GptFfn({"n_channels": config["n_channels"], "T": config["T"]})
        self.ln1 = LN({"eps": config["eps"]})
        self.ln2 = LN({"eps": config["eps"]})
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
    class Config(TypedDict):
        eps: float
        n_channels: int
        n_heads: int
        T: int
        n_blocks: int

    class Weights(TypedDict):
        blocks: List[GptBlock.Weights]

    def __init__(self, config: GptDecoder.Config):
        self.T = config["T"]
        self.n_channels = config["n_channels"]
        self.blocks = [GptBlock({"eps": config["eps"],
                                 "n_channels": config["n_channels"],
                                 "n_heads": config["n_heads"],
                                 "T": config["T"]})
                       for _ in range(config["n_blocks"])]

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
    class Config(TypedDict):
        eps: float
        n_channels: int
        n_heads: int
        T: int
        n_blocks: int
        n_tokens: int
        n_vocab: int

    class Weights(TypedDict):
        te: Arr
        decoder: GptDecoder.Weights
        ln: LN.Weights

    def __init__(self, config: Gpt.Config):
        self.T = config["T"]
        self.n_channels = config["n_channels"]
        self.n_tokens = config["n_tokens"]
        self.eps = config["eps"]
        self.pe = jnp.zeros((config["T"], config["n_channels"]))
        self.decoder = GptDecoder({"eps": config["eps"],
                                   "n_channels": config["n_channels"],
                                   "n_heads": config["n_heads"],
                                   "T": config["T"],
                                   "n_blocks": config["n_blocks"]})

        self.ln = LN({"eps": config["eps"]})

        self.lnf = for_all_T(self.ln.f)

    def init_params(self):
        return Gpt.Weights(
            te=jax.random.normal(jax.random.PRNGKey(0), (self.n_tokens, self.n_channels)),
            decoder=self.decoder.init_params(),
            ln=self.ln.init_params(),
        )

    def f(self, w: Gpt.Weights, x: Arr) -> Arr:
        assert_shape(x, (self.T,))
        x = w['te'][x] + self.pe[self.T]
        result = self.lnf(w['ln'], self.decoder.f(w['decoder'], x))
        assert_shape(result, (self.T, self.n_channels))
        return result


def gpt_loss(gpt: Gpt, inputs: list[int], labels: list[int]) -> Arr:
    logits = go(gpt, (jnp.array(inputs)))
    return softmax_cross_entropy_with_integer_labels(logits, jnp.array(labels))


def go(c, x: Arr) -> Arr:
    return c.f(c.init_params(), x)


k = go(GptMha(GptMha.Config(n_channels=9,
                            n_heads=3,
                            T=5)), jnp.ones((5, 9)))

print(k.shape)
gpt_ = Gpt(Gpt.Config(eps=1e-5,
                      n_channels=9,
                      n_heads=3,
                      T=5,
                      n_blocks=2,
                      n_tokens=10,
                      n_vocab=10))

zz = go(gpt_, jnp.ones((5,), dtype=jnp.int32))

print(zz.shape)
print(gpt_loss(gpt_, [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]))
# %%
