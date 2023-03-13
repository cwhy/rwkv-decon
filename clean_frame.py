# %%
from __future__ import annotations

import math
from abc import abstractmethod
from typing import NamedTuple, TypeVar, Callable, List, TypedDict, Union, Literal, Final, Protocol, Mapping, cast

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


ArrayGen = Literal['kaiming', 'dropout', 'embedding', 'normal']


class WeightConfig(NamedTuple):
    # from in to out
    name: str
    shape: tuple[int, ...]
    init: Union[ArrayGen, int, float] = "kaiming"
    scale: float = 1


class MakeLinear(NamedTuple):
    n_in: int
    n_out: int
    w_name: str = "w"
    b_name: str = "b"
    w_init: Literal["kaiming"] = "kaiming"
    b_init: Literal[0] = 0

    def make(self) -> Linear:
        return Linear(self)


WeightConfigTree = Mapping[str, Union[WeightConfig, "WeightConfigTree", list["WeightConfigTree"]]]
WeightsTree = dict[str, Union[Arr, "WeightsTree", list["WeightsTree"]]]


class NNModule(Protocol):
    def f(self, w: W, x: Arr) -> Arr:
        ...

    @abstractmethod
    @property
    def weight_configs(self) -> WeightConfigTree:
        ...


class Linear:
    class Weights(TypedDict):
        w: Arr
        b: Arr

    class MakeWeights(TypedDict):
        w: WeightConfig
        b: WeightConfig

    def __init__(self, config: MakeLinear):
        self.config = config
        self.n_in = config.n_in
        self.n_out = config.n_out

        self.weight_configs = self.MakeWeights(
            w=WeightConfig(
                name=config.w_name,
                shape=(self.n_out, self.n_in),
                init=config.w_init
            ),
            b=WeightConfig(
                name=config.b_name,
                shape=(self.n_out,),
                init=config.b_init)
        )

    def init_params(self):
        return self.Weights(
            w=jnp.zeros((self.n_out, self.n_in)),
            b=jnp.zeros((self.n_out,))
        )

    def f(self, w: Weights, x: Arr) -> Arr:
        assert_shape(x, (self.n_in,))
        result = w['w'] @ x + w['b']
        assert_shape(result, (self.n_out,))
        return result


class MakeGptMha(NamedTuple):
    n_channels: int
    n_heads: int
    T: int
    QKV_name: str = "QKV"
    QKV_init: Literal["normal"] = "normal"
    QKV_scale: float = 0.02

    def make(self) -> GptMha:
        return GptMha(self)


class GptMha:
    class Weights(TypedDict):
        QKV: Arr
        linear: Linear.Weights

    class MakeWeights(TypedDict):
        QKV: WeightConfig
        linear: Linear.MakeWeights

    def causal_dot_attention(self, q: Arr, k: Arr, v: Arr) -> Arr:
        assert_shape([q, k, v], (self.dim_heads, self.T))
        result = softmax((q.T @ k) / self.scale + self.mask) @ v.T
        assert_shape(result, (self.T, self.dim_heads))
        return result

    def __init__(self, config: MakeGptMha):
        self.n_channels = config.n_channels
        self.n_heads = config.n_heads
        self.T = config.T

        self.linear = MakeLinear(self.n_channels, self.n_channels).make()
        self.scale = math.sqrt(self.n_channels)
        self.mask = jnp.tril(jnp.ones((self.T, self.T)))
        self.linearf = for_all_T(self.linear.f)
        assert self.n_channels % self.n_heads == 0, 'n_channels must be divisible by n_heads'
        self.dim_heads: int = self.n_channels // self.n_heads

        self.weight_configs = self.MakeWeights(
            QKV=WeightConfig(
                name=config.QKV_name,
                shape=(3, self.n_heads, self.dim_heads, self.n_channels),
                init=config.QKV_init,
                scale=config.QKV_scale
            ),
            linear=self.linear.weight_configs
        )

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


class MakeLN(NamedTuple):
    eps: float
    w_name: str = "w"
    b_name: str = "b"
    w_init: Literal[0] = 0
    b_init: Literal[0] = 0

    def make(self) -> LN:
        return LN(self)


class LN:
    class Weights(TypedDict):
        w: Arr
        b: Arr

    class MakeWeights(TypedDict):
        w: WeightConfig
        b: WeightConfig

    def __init__(self, config: MakeLN):
        self.eps = config.eps
        self.weight_configs = self.MakeWeights(
            w=WeightConfig(
                name=config.w_name,
                shape=(1,),
                init=config.w_init
            ),
            b=WeightConfig(
                name=config.b_name,
                shape=(1,),
                init=config.b_init
            )
        )

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


class MakeGptFfn(NamedTuple):
    n_channels: int
    T: int

    def make(self) -> GptFfn:
        return GptFfn(self)


class GptFfn:
    class Weights(TypedDict):
        linear1: Linear.Weights
        linear2: Linear.Weights

    class MakeWeights(TypedDict):
        linear1: Linear.MakeWeights
        linear2: Linear.MakeWeights

    def __init__(self, config: MakeGptFfn):
        self.n_channels = config.n_channels
        self.linear1 = MakeLinear(self.n_channels, self.n_channels).make()
        self.linear2 = MakeLinear(self.n_channels, self.n_channels).make()
        self.weight_configs = self.MakeWeights(
            linear1=self.linear1.weight_configs,
            linear2=self.linear2.weight_configs
        )

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


class MakeGptBlock(NamedTuple):
    eps: float
    n_channels: int
    n_heads: int
    T: int

    def make(self) -> GptBlock:
        return GptBlock(self)


class GptBlock:
    class Weights(TypedDict):
        mha: GptMha.Weights
        ffn: GptFfn.Weights
        ln1: LN.Weights
        ln2: LN.Weights

    class MakeWeights(TypedDict):
        mha: GptMha.MakeWeights
        ffn: GptFfn.MakeWeights
        ln1: LN.MakeWeights
        ln2: LN.MakeWeights

    def __init__(self, config: MakeGptBlock):
        self.T = config.T
        self.n_channels = config.n_channels
        self.mha = MakeGptMha(config.n_channels, config.n_heads, config.T).make()
        self.ffn = MakeGptFfn(config.n_channels, config.T).make()
        self.ln1 = MakeLN(config.eps).make()
        self.ln2 = MakeLN(config.eps).make()

        self.ffnf = for_all_T(self.ffn.f)
        self.ln1f = for_all_T(self.ln1.f)
        self.ln2f = for_all_T(self.ln2.f)

        self.weight_configs = self.MakeWeights(
            mha=self.mha.weight_configs,
            ffn=self.ffn.weight_configs,
            ln1=self.ln1.weight_configs,
            ln2=self.ln2.weight_configs
        )

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


class MakeGptDecoder(NamedTuple):
    eps: float
    n_channels: int
    n_heads: int
    T: int
    n_blocks: int

    def make(self) -> GptDecoder:
        return GptDecoder(self)


class GptDecoder:
    class Weights(TypedDict):
        blocks: List[GptBlock.Weights]

    class MakeWeights(TypedDict):
        blocks: List[GptBlock.MakeWeights]

    def __init__(self, config: MakeGptDecoder):
        self.T = config.T
        self.n_channels = config.n_channels
        self.blocks = [MakeGptBlock(config.eps, config.n_channels, config.n_heads, config.T).make()
                       for _ in range(config.n_blocks)]
        self.weight_configs = self.MakeWeights(
            blocks=[blk.weight_configs for blk in self.blocks]
        )

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


class MakeGpt(NamedTuple):
    eps: float
    n_channels: int
    n_heads: int
    T: int
    n_blocks: int
    n_tokens: int
    n_vocab: int

    te_name: str = 'te'
    te_init: Literal['normal'] = 'normal'
    te_scale: float = 0.02

    def make(self) -> Gpt:
        return Gpt(self)


class Gpt:
    class Weights(TypedDict):
        te: Arr
        decoder: GptDecoder.Weights
        ln: LN.Weights

    class MakeWeights(TypedDict):
        te: WeightConfig
        decoder: GptDecoder.MakeWeights
        ln: LN.MakeWeights

    def __init__(self, config: MakeGpt):
        self.T = config.T
        self.n_channels = config.n_channels
        self.n_tokens = config.n_tokens
        self.eps = config.eps
        self.pe = jnp.zeros((config.T, config.n_channels))
        self.decoder = MakeGptDecoder(config.eps, config.n_channels, config.n_heads, config.T, config.n_blocks).make()
        self.ln = MakeLN(config.eps).make()

        self.lnf = for_all_T(self.ln.f)

        self.weight_configs = self.MakeWeights(
            te=WeightConfig(config.te_name,
                            (config.n_tokens, config.n_channels),
                            config.te_init, config.te_scale),
            decoder=self.decoder.weight_configs,
            ln=self.ln.weight_configs
        )

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


def load_weights(weight_configs: WeightConfigTree, weights_dict: dict[str, Arr], prefix: str = "") -> WeightsTree:
    weights: WeightsTree = {}
    for name, weight_config in weight_configs.items():
        if isinstance(weight_config, WeightConfig):
            weights[name] = weights_dict[prefix + weight_config.name]
        else:
            prefix += "." if prefix else ""
            prefix += name + "." if name else ""
            if isinstance(weight_config, dict):
                weight_config = cast(WeightConfigTree, weight_config)
                weights[name] = load_weights(weight_config, weights_dict, prefix)
            else:
                weight_config = cast(list[WeightConfigTree], weight_config)
                weights[name] = [load_weights(wc, weights_dict, prefix) for wc in weight_config]
    return weights



def go(c, x: Arr) -> Arr:
    return c.f(c.init_params(), x)


k = go(MakeGptMha(n_channels=9,
                  n_heads=3,
                  T=5).make(), jnp.ones((5, 9)))

print(k.shape)
gpt_ = MakeGpt(eps=1e-5,
               n_channels=9,
               n_heads=3,
               T=5,
               n_blocks=2,
               n_tokens=10,
               n_vocab=10).make()
zz = go(gpt_, jnp.ones((5,), dtype=jnp.int32))

print(zz.shape)
print(gpt_loss(gpt_, [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]))
# %%
w = gpt_.init_params()
print(w)
