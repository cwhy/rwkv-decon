from functools import partial

import jax.numpy as np
from jax import jit
from jax.lax import rsqrt
from jax.nn import sigmoid, relu


def layer_norm(x, weight, bias, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) * rsqrt(variance + eps) + bias


def time_mix(x, old_state, mix):
    return x * mix + old_state * (1 - mix)


def exp_mix(max_coef, k, v_s, v, base):
    """
    1. If `max_coef >= k`, the mixed value and the updated base value are computed as follows:
       mix_v = v_s + e^(k - max_coef) * v
       base_new = base + e^(k - max_coef)
       
    2. If `max_coef < k`, the mixed value and the updated base value are computed as follows:
       mix_v = e^(max_coef - k) * v_s + v
       base_new = e^(max_coef - k) * base + 1
    """
    new_max_coef = np.maximum(max_coef, k)
    e1 = np.exp(max_coef - new_max_coef)
    e2 = np.exp(k - new_max_coef)
    mix_v = e1 * v_s + e2 * v
    base_new = e1 * base + e2
    return mix_v, base_new, new_max_coef


def rwkv_state_flow(k, v, state_wkv, time_decay):
    wkv_top, wkv_bot, max_coef = state_wkv
    return np.stack(exp_mix(max_coef + time_decay, k, wkv_top, v, wkv_bot))


def rwkv(r, ow, k, v, state_wkv, time_first):
    """
    state_wkv: (wkv_top, wkv_bot, max_coef) / state[i, 2:].T
    """
    wkv_top, wkv_bot, max_coef = state_wkv

    wkv_top, wkv_bot, _ = exp_mix(max_coef, k + time_first, wkv_top, v, wkv_bot)
    wkv = wkv_top / wkv_bot
    return ow @ (r * wkv)


def token_mixing(x, token_mix_state, time_mix_k, time_mix_v, time_mix_r, kw, vw, rw, **kwargs):
    xk = time_mix(x, token_mix_state, time_mix_k)
    xv = time_mix(x, token_mix_state, time_mix_v)
    xr = time_mix(x, token_mix_state, time_mix_r)

    r = sigmoid(rw @ xr)
    k = kw @ xk
    v = vw @ xv
    return r, k, v


def channel_mixing(x, channel_mix_state, time_mix_k, time_mix_r, kw, vw, rw, **kwargs):
    xk = time_mix(x, channel_mix_state, time_mix_k)
    xr = time_mix(x, channel_mix_state, time_mix_r)
    r = sigmoid(rw @ xr)
    k = np.square(relu(kw @ xk))  # square relu, primer paper
    return r * (vw @ k)


def rwkv_net(token, state, ln_out, blocks, head, emb):
    x = emb['weight'][token]
    w_ln0 = blocks[0]['ln0']
    x = layer_norm(x, **w_ln0)
    new_states = []
    for i in range(len(blocks)):
        block_w = blocks[i]

        xn_token = layer_norm(x, **blocks[i]['ln1'])
        r, k, v = token_mixing(xn_token, state[i, 1], **block_w['att'])

        state_wkv = state[i, 2:]

        x += rwkv(r, block_w['att']['output']['weight'], k, v, state_wkv, block_w['att']['time_first'])
        xn_channel = layer_norm(x, **blocks[i]['ln2'])
        xp = channel_mixing(xn_channel, state[i, 0], **block_w['ffn'])

        x += xp
        new_state = np.vstack([xn_channel, xn_token, rwkv_state_flow(k, v, state_wkv, block_w['att']['time_decay'])])
        new_states.append(new_state)
    state = np.stack(new_states)
    xn = layer_norm(x, **ln_out)
    x = head['weight'] @ xn
    return x, state


@jit
def rwkv_net_w(token, state, w):
    return rwkv_net(token, state, w['ln_out'], w['blocks'], w['head'], w['emb'])

# [Done] load params
# [Done] jit
# [TODO] associative scan
# [TODO] pico-plus
