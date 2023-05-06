from copy_init.weights import WeightConfig, WeightConfigType, NormalWeight, ZeroWeight
from labels import L


def make_weight_config(name: str, shape: tuple[int, ...], wtype: WeightConfigType) -> WeightConfig:
    tags = L(*name.split('.'))
    if wtype == 'normal':
        return NormalWeight(name=name, shape=shape, tags=tags)
    elif wtype == 'zero':
        return ZeroWeight(name=name, shape=shape, tags=tags)
    else:
        raise ValueError(f"weight_config_type must be 'normal' or 'zero', not {wtype}")


def make_rwkv_weight_configs(n_blocks: int, n_embd: int, ffn_hidden_multiplier: int) -> dict[str, WeightConfig]:
    properties = [
        ("blocks.att.output.weight", (n_blocks, n_embd, n_embd), 'zero'),
        ("blocks.att.value.weight", (n_blocks, n_embd, n_embd), 'zero'),
        ("blocks.att.key.weight", (n_blocks, n_embd, n_embd), 'zero'),
        ("blocks.att.receptance.weight", (n_blocks, n_embd, n_embd), 'zero'),
        ("blocks.att.time_decay", (n_blocks, n_embd), 'zero'),
        ("blocks.att.time_first", (n_blocks, n_embd), 'zero'),
        ("blocks.att.time_mix_k", (n_blocks, n_embd), 'zero'),
        ("blocks.att.time_mix_r", (n_blocks, n_embd), 'zero'),
        ("blocks.att.time_mix_v", (n_blocks, n_embd), 'zero'),
        ("blocks.ffn.key.weight", (n_blocks, ffn_hidden_multiplier * n_embd, n_embd), 'zero'),
        ("blocks.ffn.value.weight", (n_blocks, n_embd, n_embd * ffn_hidden_multiplier), 'zero'),
        ("blocks.ffn.receptance.weight", (n_blocks, n_embd, n_embd), 'zero'),
        ("blocks.ffn.time_mix_k", (n_blocks, n_embd), 'zero'),
        ("blocks.ffn.time_mix_r", (n_blocks, n_embd), 'zero'),
        ("blocks.ffn.time_mix_v", (n_blocks, n_embd), 'zero'),
        ("blocks.ln0.weight", (n_blocks, n_embd), 'zero'),
        ("blocks.ln0.bias", (n_blocks, n_embd), 'zero'),


    ]
