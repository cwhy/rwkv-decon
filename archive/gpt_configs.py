from __future__ import annotations

import math
from functools import cache
from typing import NamedTuple, Optional, cast, TypedDict, List, Literal, Union

from chex import assert_shape
from einops import rearrange
from jax import numpy as jnp
from jax.nn import softmax
from jax.experimental.maps import xmap
from safetensors.flax import save_file

from clean_frame import Linear, for_all_T, gelu, LN
from clean_frame_utils import check_config, WeightConfigDict, PartsDict, WeightsTree, config_weights_check, Arr, \
    WeightConfig


class GptMha:
    class Config(NamedTuple):
        n_channels: Optional[int] = None
        n_heads: Optional[int] = None
        n_seq: Optional[Union[int, Literal['dynamic']]] = None
        inf_mask: float = -1e10
        linear: Linear.Config = Linear.Config()
        QKV_linear: Linear.Config = Linear.Config()

        name: str = "mha"

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

    class Weights(TypedDict):
        QKV_linear: Linear.Weights
        linear: Linear.Weights

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
    class Config(NamedTuple):
        n_channels: Optional[int] = None
        n_seq: Optional[Union[int, Literal['dynamic']]] = None
        linear1: Linear.Config = Linear.Config()
        linear2: Linear.Config = Linear.Config()
        name: str = "ffn"

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

    class Weights(TypedDict):
        linear1: Linear.Weights
        linear2: Linear.Weights

    def __init__(self, config: Config):
        self.n_channels = config.n_channels
        self.linear1 = config.linear1.make()
        self.linear2 = config.linear2.make()


class GptBlock:
    class Config(NamedTuple):
        eps: Optional[float] = None
        n_channels: Optional[int] = None
        n_heads: Optional[int] = None
        n_seq: Optional[Union[int, Literal['dynamic']]] = None
        mha: GptMha.Config = GptMha.Config()
        ffn: GptFfn.Config = GptFfn.Config()
        ln1: LN.Config = LN.Config()
        ln2: LN.Config = LN.Config()
        name: str = "gpt_block"

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
        self.ln1f = self.ln1.f
        self.ln2f = self.ln2.f



class GptDecoder:
    class Config(NamedTuple):
        eps: Optional[float] = None
        n_channels: Optional[int] = None
        n_heads: Optional[int] = None
        n_seq: Optional[Union[int, Literal['dynamic']]] = None
        n_blocks: Optional[int] = None
        blocks: GptBlock.Config = GptBlock.Config()

        name: str = 'gpt_decoder'

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

    class Weights(TypedDict):
        blocks: List[GptBlock.Weights]

    def __init__(self, config: Config):
        assert config.n_blocks is not None
        self.T = config.T
        self.n_channels = config.n_channels
        self.blocks = [config.blocks.make() for _ in range(config.n_blocks)]



class Gpt:
    class Config(NamedTuple):
        eps: Optional[float] = None
        n_channels: Optional[int] = None
        n_heads: Optional[int] = None
        n_seq: Optional[Union[int, Literal['dynamic']]] = None
        n_blocks: Optional[int] = None
        n_tokens: Optional[int] = None
        max_seq_len: Optional[int] = None

        te_name: str = 'te'
        te_init: Literal['normal'] = 'normal'
        te_scale: float = 0.02

        decoder: GptDecoder.Config = GptDecoder.Config()
        ln: LN.Config = LN.Config()

        pe_name: str = 'pe'

        name: str = 'gpt'

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
                token_embedding=WeightConfig(name=filled.te_name,
                                             init=filled.te_init,
                                             shape=(filled.n_tokens, filled.n_channels),
                                             scale=filled.te_scale),

                positional_encoding=WeightConfig(name=filled.pe_name,
                                                 shape=(filled.max_seq_len, filled.n_channels)),
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
            return cast(Gpt.Weights, config_weights_check(self, w))

    def __init__(self, config: Config):
        assert config.n_blocks is not None
        assert config.n_tokens is not None
        assert config.n_channels is not None
        self.T = config.T
        self.n_channels = config.n_channels
        self.n_tokens = config.n_tokens
        self.eps = config.eps
        self.decoder = config.decoder.make()
        self.ln = config.ln.make()

