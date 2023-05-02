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


@partial(jit, static_argnames='i')
def rwkv(r, ow, k, v, state, i: int, time_first, time_decay, debug=False):
    """
    the original form of the equation is:
    $\omega = \frac{a_1 + e^{(t_1 + k) - p} \cdot v}{b_1 + e^{(t_1 + k) - p}}$
    for numerical stability, we rewrite it as:
    $\omega = \frac{e^{p - \max(p, t_1 + k)} \cdot a_1 + e^{(t_1 + k) - \max(p, t_1 + k)} \cdot v}{e^{p - \max(p, t_1 + k)} \cdot b_1 + e^{(t_1 + k) - \max(p, t_1 + k)}}$
    where
    $a_1$ is wkv_top (state[i, 2])
    $b_1$ is wkv_bot (state[i, 3])
    $p$ is max_coef (state[i, 4])
    $\omega$ is wkv
    $t_1$ is time_first
    $t_2$ is time_decay

    :param r: [1024, ]
    :param ow:
    :param k: [1024, ]
    :param v: [1024, ]
    :param state: [50, 1024]
    :param i: int
    :param time_first: [1024, ]
    :param time_decay:
    :return: 
    """
    v_state, base_state, max_coef = state[i, 2], state[i, 3], state[i, 4]

    wkv_top, wkv_bot, _ = exp_mix(max_coef, k + time_first, v_state, v, base_state)
    wkv = wkv_top / wkv_bot
    wkv_top_new, wkv_bot_new, max_coef = exp_mix(max_coef + time_decay, k, v_state, v, base_state)


    state = state.at[i, 2].set(wkv_top_new)
    state = state.at[i, 3].set(wkv_bot_new)
    state = state.at[i, 4].set(max_coef)
    return ow @ (r * wkv), state


@partial(jit, static_argnames='i')
def token_mixing(x, state, i: int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow, debug=False):
    xk = time_mix(x, state[i, 1], time_mix_k)
    xv = time_mix(x, state[i, 1], time_mix_v)
    xr = time_mix(x, state[i, 1], time_mix_r)

    state = state.at[i, 1].set(x)

    r = sigmoid(rw @ xr)
    k = kw @ xk
    v = vw @ xv
    return rwkv(r, ow, k, v, state, i, time_first, time_decay, debug=debug)


@partial(jit, static_argnames='i')
def channel_mixing(x, state, i: int, time_mix_k, time_mix_r, kw, vw, rw):
    xk = time_mix(x, state[i, 0], time_mix_k)
    xr = time_mix(x, state[i, 0], time_mix_r)
    state = state.at[i, 0].set(x)
    r = sigmoid(rw @ xr)
    k = np.square(relu(kw @ xk))  # square relu, primer paper
    return r * (vw @ k), state


@partial(jit, static_argnames='i')
def block(x, state, i: int, att, ffn, ln1, ln2, **kwargs):
    xn = layer_norm(x, **ln1)
    xp, state = token_mixing(xn, state, i,
                             att['time_mix_k'],
                             att['time_mix_v'],
                             att['time_mix_r'],
                             att['time_first'],
                             att['time_decay'],
                             att['key']['weight'],
                             att['value']['weight'],
                             att['receptance']['weight'],
                             att['output']['weight'], debug=kwargs['debug'])

    x += xp
    xn = layer_norm(x, **ln2)
    xp, state = channel_mixing(xn, state, i, ffn['time_mix_k'], ffn['time_mix_r'],
                               ffn['key']['weight'], ffn['value']['weight'],
                               ffn['receptance']['weight'])
    x += xp
    return x, state


def rwkv_net(token, state, ln_out, blocks, head, emb):
    #print(token)
    x = emb['weight'][token]
    w_ln0 = blocks[0]['ln0']
    x = layer_norm(x, **w_ln0)
    for i in range(len(blocks)):
        block_w = blocks[i]
        x, state = block(x, state, i, debug=token == 49, **block_w)
    xn = layer_norm(x, **ln_out)
    x = head['weight'] @ xn
    return x, state


@jit
def rwkv_net_w(token, state, w):
    return rwkv_net(token, state, w['ln_out'], w['blocks'], w['head'], w['emb'])

