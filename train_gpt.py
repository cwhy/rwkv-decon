# %%
from __future__ import annotations

from pprint import pprint

import jax.numpy as jnp
from jax import random
from optax import softmax_cross_entropy_with_integer_labels

from clean_frame_utils import Arr, WeightConfigDict, PartsDict, WeightsTree, ModuleConfig, WeightConfig, \
    config_weights_check
from gpt import GptMha, Gpt
from jax_init_utils import RNGKey


def gpt_loss(gpt: Gpt, inputs: list[int], labels: list[int]) -> Arr:
    logits = go(gpt, (jnp.array(inputs)))
    return softmax_cross_entropy_with_integer_labels(logits, jnp.array(labels))


def go(c, x: Arr) -> Arr:
    weights = init_weights(c)
    return c.f(weights, x)


gpt_mha_config_ = GptMha.Config(n_channels=9,
                                n_heads=3,
                                n_seq='dynamic').fill()
pprint(gpt_mha_config_.parts)


# TODO RNG spreading
def init_weights(weights_config: WeightConfigDict, rng_key: RNGKey) -> WeightsTree:
    w: WeightsTree = {}
    for name, weight_config in weights_config.items():
        assert isinstance(weight_config, WeightConfig), f"Config {name} should be a WeightConfig, got {weight_config}"
        w[name] = weight_config.make(rng_key)
    return w


# TODO RNG spreading
def init_weight_module(module: ModuleConfig, rng_key: RNGKey) -> WeightsTree:
    try:
        w = init_weight_parts(module.parts, rng_key)
        w.update(init_weights(module.weights, rng_key))
    except Exception as e:
        raise Exception(f"Failed to initialize module {module.name}.") from e
    return w


# TODO RNG spreading
def init_weight_parts(parts: PartsDict, rng_key: RNGKey) -> WeightsTree:
    w: WeightsTree = {}
    for name, part_config in parts.items():
        if isinstance(part_config, ModuleConfig):
            w[name] = init_weight_module(part_config, rng_key)
        else:
            err_msg = f"Config {name} should be a ModuleConfig or a non-empty list of ModuleConfigs, got {part_config}"
            assert isinstance(part_config, list), err_msg
            assert len(part_config) > 0, err_msg
            assert isinstance(part_config[0], ModuleConfig), err_msg
            w[name] = [init_weight_module(part, rng_key) for i, part in enumerate(part_config)]
    return w


key = random.PRNGKey(0)
w = init_weight_parts(gpt_mha_config_.parts, key)
print(w)
checked = config_weights_check(gpt_mha_config_, w)
print(checked)

# %%
gpt_mha_ = gpt_mha_config_.make(), jnp.ones((5, 9))

print(go(gpt_mha_).shape)
gpt_ = Gpt.Config(eps=1e-5,
                  n_channels=9,
                  n_heads=3,
                  n_seq='dynamic',
                  n_blocks=2,
                  n_tokens=10).make()
zz = go(gpt_, jnp.ones((5,), dtype=jnp.int32))

print(zz.shape)
print(gpt_loss(gpt_, [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]))
www = gpt_.init_params()
