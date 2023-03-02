#%%
from __future__ import annotations

import math
from typing import Callable

import jax
from jax.experimental.maps import xmap
from optax import softmax_cross_entropy_with_integer_labels
from simple_pytree import Pytree, static_field
from jax.numpy import mean, var, sqrt, tanh, pi
from jax.nn import softmax
from jax.lax import rsqrt
import jax.numpy as jnp

Arr = jax.Array


def gelu(x: Arr) -> Arr:
    return 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x ** 3)))


class LN(Pytree):
    eps: float = static_field()

    def __init__(self, eps: float, w: Arr, b: Arr):
        self.eps = eps
        self.w = w
        self.b = b

    # x: (n_channels,)
    def f(self, x: Arr) -> Arr:
        o = x - mean(x)
        i = self.w * rsqrt(var(x) + self.eps)
        return o * i + self.b


class Linear(Pytree):
    def __init__(self, w: Arr, b: Arr):
        self.w = w
        self.b = b

    # x: (n_channels,)
    def f(self, x: Arr) -> Arr:
        return self.w @ x + self.b


class GptMha(Pytree):
    n_heads: int = static_field()
    scale: float = static_field()
    dim_head: int = static_field()
    T: int = static_field()
    mask: Arr = static_field()
    linearf: Callable[[Arr], Arr] = static_field()

    # q, k, v: (dim_head, T)
    def causal_dot_attention(self, q: Arr, k: Arr, v: Arr) -> Arr:
        return softmax((q.T @ k) / self.scale + self.mask) @ v

    # QKV: (3, n_heads, dim_head, n_channels)
    def __init__(self, n_channels: int, n_heads: int, T: int, QKV: Arr, linear: Linear):
        self.scale = math.sqrt(n_channels)
        self.n_heads = n_heads
        self.QKV = QKV
        self.linear = linear
        self.mask = jnp.tril(jnp.ones((T, T)))
        self.linearf = xmap(linear.f, [None, 'T'], 'T')

    # x: (n_channels, T)
    def f(self, x: Arr) -> Arr:
        q, k, v = self.QKV @ x
        xmap_in = [['n_head', ...]] * 3
        attended = xmap(self.causal_dot_attention, xmap_in, 'n_head')(q, k, v)
        # attended: (n_heads, dim_head, T)

        return self.linearf(jnp.concatenate(attended))


class GptFfn(Pytree):
    def __init__(self, linear1: Linear, linear2: Linear):
        self.linear1 = linear1
        self.linear2 = linear2

    # x: (n_channels,)
    def f(self, x: Arr) -> Arr:
        return self.linear2.f(gelu(self.linear1.f(x.T))).T


class GptBlock(Pytree):
    ln1f: Callable[[Arr], Arr] = static_field()
    ln2f: Callable[[Arr], Arr] = static_field()
    ffnf: Callable[[Arr], Arr] = static_field()

    def __init__(self, mha: GptMha, ffn: GptFfn, ln1: LN, ln2: LN):
        self.mha = mha
        self.ffn = ffn
        self.ln1 = ln1
        self.ln2 = ln2
        self.ln1f = xmap(ln1.f, [None, 'T'], 'T')
        self.ln2f = xmap(ln2.f, [None, 'T'], 'T')
        self.ffnf = xmap(ffn.f, [None, 'T'], 'T')

    # x: (n_channels, T)
    def f(self, x: Arr) -> Arr:
        x += self.mha.f(self.ln1f(x))
        x += self.ffnf(self.ln2f(x))
        return x


class GptDecoder(Pytree):
    def __init__(self, blocks: list[GptBlock]):
        self.blocks = blocks

    # x: (n_channels, T)
    def f(self, x: Arr) -> Arr:
        for block in self.blocks:
            x = block.f(x)
        return x


class Gpt(Pytree):
    T: int = static_field()

    def __init__(self, T: int, decoder: GptDecoder, token_embed: Arr, position_embed: Arr, ln: LN):
        self.decoder = decoder
        self.ln = ln
        self.te = token_embed
        self.pe = position_embed
        self.T = T

    # x: (n_channels, T)
    def f(self, x: Arr) -> Arr:
        x = self.te[x] + self.pe[self.T]
        return self.ln.f(self.decoder.f(x))


def gpt_loss(gpt: Gpt, inputs: list[int], labels: list[int]) -> Arr:
    logits = gpt.f(jnp.array(inputs))
    return softmax_cross_entropy_with_integer_labels(logits, labels)


