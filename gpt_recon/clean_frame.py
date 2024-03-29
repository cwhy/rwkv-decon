# %%
from __future__ import annotations
from typing import NamedTuple, TypeVar, Callable, TypedDict, Literal, cast, \
    Optional

from chex import assert_shape
from jax import vmap
from jax.experimental.maps import xmap
from jax.lax import rsqrt
from jax.numpy import mean, var, sqrt, tanh, pi

from picojax.jax_utils import jit_f, WeightsTree
from gpt_recon.clean_frame_utils import Arr, PartsDict, WeightConfig, WeightConfigDict, check_config, config_weights_check

C = TypeVar('C')
W = TypeVar('W')


# TODO: unify init_param
# TODO: load real GPT weights


def no_w(d: C) -> tuple[list, C]:
    return [...], d


def batch_ops_x(f: Callable[[W, Arr], Arr], label: str, add_behind: bool, skip_w: bool) -> Callable[[W, Arr], Arr]:
    if add_behind:
        extension = [..., label]
    else:
        extension = [label, ...]
    if skip_w:
        return xmap(f, no_w(extension), extension)
    else:
        return xmap(f, extension, extension)


def batch_ops(f: Callable[[W, Arr], Arr], label: str, add_behind: bool, skip_w: bool) -> Callable[[W, Arr], Arr]:
    if add_behind:
        extension = -1
    else:
        extension = 0
    if skip_w:
        return vmap(f, (None, extension), extension)
    else:
        return vmap(f, extension, extension)


def for_all_T(f: Callable[[W, Arr], Arr]) -> Callable[[W, Arr], Arr]:
    return batch_ops(f, 'T', False, True)


def batch_fy(f: Callable[[W, Arr], Arr]) -> Callable[[W, Arr], Arr]:
    return batch_ops(f, 'batch', False, True)


def gelu(x: Arr) -> Arr:
    return 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x ** 3)))


class Linear:
    class Weights(TypedDict):
        w: Arr
        b: Arr

    @jit_f
    def f(self, w: Weights, x: Arr) -> Arr:
        assert_shape(x, (self.n_in,))
        result = w['w'].T @ x + w['b']
        assert_shape(result, (self.n_out,))
        return result

    class Config(NamedTuple):
        n_in: Optional[int] = None
        n_out: Optional[int] = None
        w_save_name: str = "w"
        b_save_name: str = "b"
        w_init: Literal["kaiming"] = "kaiming"
        b_init: Literal[0] = 0
        save_name: str = "linear"

        def make(self) -> Linear:
            check_config(self)
            return Linear(self)

        @property
        def weights(self) -> WeightConfigDict:
            assert self.n_in is not None
            assert self.n_out is not None
            return dict(
                w=WeightConfig(
                    save_name=self.w_save_name,
                    shape=(self.n_in, self.n_out),
                    init=self.w_init
                ),
                b=WeightConfig(
                    save_name=self.b_save_name,
                    shape=(self.n_out,),
                    init=self.b_init)
            )

        @property
        def parts(self) -> PartsDict:
            return {}

        def fill(self) -> Linear.Config:
            return self

        def weights_check(self, w: WeightsTree) -> Linear.Weights:
            return cast(Linear.Weights, config_weights_check(self, w))

    def __init__(self, config: Config):
        self.config = config
        assert config.n_in is not None
        assert config.n_out is not None
        self.n_in = config.n_in
        self.n_out = config.n_out


class LN:
    class Weights(TypedDict):
        w: Arr
        b: Arr

    # x: self.shape
    @jit_f
    def f(self, w: LN.Weights, x: Arr) -> Arr:
        o = x - mean(x, axis=self.config.non_norm_dims, keepdims=True)
        i = o * rsqrt(var(x, axis=self.config.non_norm_dims, keepdims=True) + self.eps)
        return w['w'] * i + w['b']

    class Config(NamedTuple):
        eps: Optional[float] = None
        # all the other dimensions are normalized
        norm_dims: Optional[tuple[int, ...]] = None
        x_shape: Optional[tuple[Optional[int], ...]] = None
        w_save_name: str = "w"
        b_save_name: str = "b"
        w_init: Literal[0] = 0
        b_init: Literal[0] = 0
        save_name: str = "ln"
        norm_dim_name: str = "norm_dim"

        def make(self) -> LN:
            check_config(self)
            return LN(self)

        @property
        def non_norm_dims(self) -> tuple[int, ...]:
            assert self.norm_dims is not None
            assert self.x_shape is not None
            return tuple(i for i in range(len(self.x_shape)) if i not in self.norm_dims)

        @property
        def weights(self) -> WeightConfigDict:
            assert self.norm_dims is not None, 'norm_dims must be specified'
            assert self.x_shape is not None, 'x_shape must be specified'
            assert self.eps is not None, 'eps must be specified'
            non_norm_shape = tuple(self.x_shape[i] for i in range(len(self.x_shape)) if i not in self.norm_dims)
            non_norm_shape = cast(tuple[int, ...], non_norm_shape)
            return dict(
                w=WeightConfig(
                    save_name=self.w_save_name,
                    shape=non_norm_shape,
                    init=self.w_init
                ),
                b=WeightConfig(
                    save_name=self.b_save_name,
                    shape=non_norm_shape,
                    init=self.b_init
                )
            )

        @property
        def parts(self) -> PartsDict:
            return {}

        def fill(self) -> LN.Config:
            return self

        def weights_check(self, w: WeightsTree) -> LN.Weights:
            return cast(LN.Weights, config_weights_check(self, w))

    def __init__(self, config: Config):
        assert config.eps is not None
        self.config = config
        self.eps = config.eps
