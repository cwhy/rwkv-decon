# %%
from __future__ import annotations

import math
from abc import abstractmethod
from collections import defaultdict
from typing import NamedTuple, TypeVar, Callable, List, TypedDict, Union, Literal, Protocol, Mapping, cast, \
    Optional

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


class NNModule(Protocol):
    def __init__(self, config: NamedTuple):
        ...

    def f(self, w: dict, x: Arr) -> Arr:
        ...


T = TypeVar('T')


def check_config(config: NamedTuple) -> None:
    for k, v in config._asdict().items():
        if v is None:
            raise ValueError(f"Missing config '{k}' in {config.__class__}")


WeightConfigTree = Mapping[str, Union[WeightConfig, "WeightConfigTree", list["WeightConfigTree"]]]
WeightsTree = dict[str, Union[Arr, "WeightsTree", list["WeightsTree"]]]


class Linear:
    class Config(NamedTuple):
        n_in: Optional[int] = None
        n_out: Optional[int] = None
        w_name: str = "w"
        b_name: str = "b"
        w_init: Literal["kaiming"] = "kaiming"
        b_init: Literal[0] = 0

        def make(self) -> Linear:
            check_config(self)
            return Linear(self)

        @property
        def weights(self) -> Linear.MakeWeights:
            assert self.n_in is not None
            assert self.n_out is not None
            return Linear.MakeWeights(
                w=WeightConfig(
                    name=self.w_name,
                    shape=(self.n_out, self.n_in),
                    init=self.w_init
                ),
                b=WeightConfig(
                    name=self.b_name,
                    shape=(self.n_out,),
                    init=self.b_init)
            )

    class Weights(TypedDict):
        w: Arr
        b: Arr

    class MakeWeights(TypedDict):
        w: WeightConfig
        b: WeightConfig

    def __init__(self, config: Config):
        self.config = config
        assert config.n_in is not None
        assert config.n_out is not None
        self.n_in = config.n_in
        self.n_out = config.n_out

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


class GptMha:
    class Config(NamedTuple):
        n_channels: Optional[int] = None
        n_heads: Optional[int] = None
        T: Optional[int] = None
        linear: Linear.Config = Linear.Config()
        QKV_name: str = "QKV"
        QKV_init: Literal["normal"] = "normal"
        QKV_scale: float = 0.02

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
        def weights(self) -> GptMha.MakeWeights:
            filled = self.fill()
            assert filled.n_channels is not None
            assert filled.n_heads is not None
            assert filled.dim_heads is not None
            return GptMha.MakeWeights(
                QKV=WeightConfig(
                    name=filled.QKV_name,
                    shape=(3, filled.n_heads, filled.dim_heads, filled.n_channels),
                    init=filled.QKV_init,
                    scale=filled.QKV_scale
                ),
                linear=filled.linear.weights
            )

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
        w_name: str = "w"
        b_name: str = "b"
        w_init: Literal[0] = 0
        b_init: Literal[0] = 0

        def make(self) -> LN:
            check_config(self)
            return LN(self)

        @property
        def weights(self) -> LN.MakeWeights:
            return LN.MakeWeights(
                w=WeightConfig(
                    name=self.w_name,
                    shape=(1,),
                    init=self.w_init
                ),
                b=WeightConfig(
                    name=self.b_name,
                    shape=(1,),
                    init=self.b_init
                )
            )

    class Weights(TypedDict):
        w: Arr
        b: Arr

    class MakeWeights(TypedDict):
        w: WeightConfig
        b: WeightConfig

    def __init__(self, config: Config):
        assert config.eps is not None
        self.eps = config.eps

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
    class Config(NamedTuple):
        n_channels: Optional[int] = None
        T: Optional[int] = None
        linear1: Linear.Config = Linear.Config()
        linear2: Linear.Config = Linear.Config()

        def fill(self) -> GptFfn.Config:
            new = self._replace(
                linear1=self.linear1._replace(n_in=self.n_channels, n_out=self.n_channels),
                linear2=self.linear2._replace(n_in=self.n_channels, n_out=self.n_channels))
            check_config(new)
            return new

        def make(self) -> GptFfn:
            return GptFfn(self.fill())

        @property
        def weights(self) -> GptFfn.MakeWeights:
            filled = self.fill()
            return GptFfn.MakeWeights(
                linear1=filled.linear1.weights,
                linear2=filled.linear2.weights,
            )

    class Weights(TypedDict):
        linear1: Linear.Weights
        linear2: Linear.Weights

    class MakeWeights(TypedDict):
        linear1: Linear.MakeWeights
        linear2: Linear.MakeWeights

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
        def weights(self) -> GptBlock.MakeWeights:
            filled = self.fill()
            return GptBlock.MakeWeights(
                mha=filled.mha.weights,
                ffn=filled.ffn.weights,
                ln1=filled.ln1.weights,
                ln2=filled.ln2.weights,
            )

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

        name: str = 'blocks'

        def fill(self) -> GptDecoder.Config:
            new = self._replace(blocks=self.blocks._replace(eps=self.eps, n_channels=self.n_channels,
                                                            n_heads=self.n_heads, T=self.T).fill())
            check_config(new)
            return new

        def make(self) -> GptDecoder:
            return GptDecoder(self.fill())

        @property
        def weights(self) -> GptDecoder.MakeWeights:
            filled = self.fill()
            assert filled.blocks is not None
            assert filled.n_blocks is not None
            return GptDecoder.MakeWeights(
                blocks=[filled.blocks.weights] * filled.n_blocks
            )

    class Weights(TypedDict):
        blocks: List[GptBlock.Weights]

    class MakeWeights(TypedDict):
        blocks: List[GptBlock.MakeWeights]

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
        n_vocab: Optional[int] = None

        te_name: str = 'te'
        te_init: Literal['normal'] = 'normal'
        te_scale: float = 0.02

        decoder: GptDecoder.Config = GptDecoder.Config()
        ln: LN.Config = LN.Config()

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
        def weights(self) -> Gpt.MakeWeights:
            filled = self.fill()
            assert filled.decoder is not None
            assert filled.ln is not None
            assert filled.n_tokens is not None
            assert filled.n_channels is not None
            return Gpt.MakeWeights(
                te=WeightConfig(name=filled.te_name,
                                init=filled.te_init,
                                shape=(filled.n_tokens, filled.n_channels),
                                scale=filled.te_scale),
                decoder=filled.decoder.weights,
                ln=filled.ln.weights
            )

    class Weights(TypedDict):
        te: Arr
        decoder: GptDecoder.Weights
        ln: LN.Weights

    class MakeWeights(TypedDict):
        te: WeightConfig
        decoder: GptDecoder.MakeWeights
        ln: LN.MakeWeights

    def __init__(self, config: Config):
        assert config.n_blocks is not None
        assert config.n_tokens is not None
        assert config.n_vocab is not None
        assert config.n_channels is not None
        self.T = config.T
        self.n_channels = config.n_channels
        self.n_tokens = config.n_tokens
        self.eps = config.eps
        self.pe = jnp.zeros((config.T, config.n_channels))
        self.decoder = config.decoder.make()
        self.ln = config.ln._replace(eps=config.eps).make()

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


kkk = go(GptMha.Config(n_channels=9,
                       n_heads=3,
                       T=5).make(), jnp.ones((5, 9)))

print(kkk.shape)
gpt_ = Gpt.Config(eps=1e-5,
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
www = gpt_.init_params()
# print(www)

gpt_config = Gpt.Config(eps=1e-5,
                        n_channels=9,
                        n_heads=3,
                        T=5,
                        n_blocks=2,
                        n_tokens=10,
                        n_vocab=10,
                        te_name='wte.weight',
                        ln=LN.Config(w_name='ln_f.weight', b_name='ln_f.bias'),
                        decoder=GptDecoder.Config(
                            name='h',
                            blocks=GptBlock.Config(
                                mha=GptMha.Config(
                                    QKV_name='attn.c_attn.weight',
                                    linear=Linear.Config(w_name='attn.c_proj.weight',
                                                         b_name='attn.c_proj.bias')
                                ),
                                ln1=LN.Config(w_name='ln_1.weight', b_name='ln_1.bias'),
                                ffn=GptFfn.Config(
                                    linear1=Linear.Config(w_name='mlp.c_fc.weight',
                                                          b_name='mlp.c_fc.bias'),
                                    linear2=Linear.Config(w_name='mlp.c_proj.weight',
                                                          b_name='mlp.c_proj.bias')
                                ),
                                ln2=LN.Config(w_name='ln_2.weight', b_name='ln_2.bias')
                            )
                        )).fill()

print(gpt_config)
print(gpt_config.weights)


# %%
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

z = defaultdict(int)
ww = load_weights(gpt_config.weights, z)
print(z.keys())

# todo: separate weights from components for weight config