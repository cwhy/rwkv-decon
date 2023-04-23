from functools import partial

import jax
import jax.numpy as np
from jax import jit, vmap
from jax.lax import rsqrt
from jax.nn import sigmoid, relu


def layer_norm(x, weight, bias, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) * rsqrt(variance + eps) + bias


def time_mix(x, x_prev, mix):
    return x * mix + x_prev * (1 - mix)


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


def rwkv(r, ow, k, v, state_wkv, time_first):
    """
    state_wkv: (v_state, wkv_bot, max_coef) / state[i, 2:].T
    """
    v_state, base_state, max_coef = state_wkv

    v_state, base_state, _ = exp_mix(max_coef, k + time_first, v_state, v, base_state)
    wkv = v_state / base_state
    return ow @ (r * wkv)


def rwkv_state_flow(time_decay, state_wkv, k, v):
    v_state, base_state, max_coef = state_wkv
    return exp_mix(max_coef + time_decay, k, v_state, v, base_state)


def token_mixing(x, x_prev, time_mix_k, time_mix_v, time_mix_r, kw, vw, rw, **kwargs):
    xk = time_mix(x, x_prev, time_mix_k)
    xv = time_mix(x, x_prev, time_mix_v)
    xr = time_mix(x, x_prev, time_mix_r)

    r = sigmoid(rw @ xr)
    k = kw @ xk
    v = vw @ xv
    return r, k, v


def channel_mixing(x, x_prev, time_mix_k, time_mix_r, kw, vw, rw, **kwargs):
    xk = time_mix(x, x_prev, time_mix_k)
    xr = time_mix(x, x_prev, time_mix_r)
    r = sigmoid(rw @ xr)
    k = np.square(relu(kw @ xk))  # square relu, primer paper
    return r * (vw @ k)


def token_mixing_parallel(x, x_prev1, time_mix_k, time_mix_v, time_mix_r, kw, vw, rw, **kwargs):
    x_prev = np.vstack((x_prev1, x[:-1, :]))
    token_mixing_p = vmap(token_mixing, in_axes=(0, 0, None, None, None, None, None, None), out_axes=(0, 0, 0))
    return token_mixing_p(x, x_prev, time_mix_k, time_mix_v, time_mix_r, kw, vw, rw)


def channel_mixing_parallel(x, x_prev1, time_mix_k, time_mix_r, kw, vw, rw, **kwargs):
    x_prev = np.vstack((x_prev1, x[:-1, :]))
    channel_mixing_p = vmap(channel_mixing, in_axes=(0, 0, None, None, None, None, None), out_axes=0)
    return channel_mixing_p(x, x_prev, time_mix_k, time_mix_r, kw, vw, rw)


def rwkv_state_flow_scan(time_decay, state_wkv, kv):
    # state_wkv.shape: t, 3, n_channels
    k, v = kv
    new_state = rwkv_state_flow(time_decay, state_wkv, k, v)
    return np.stack(new_state), (k, v)


@jit
def rwkv_net_rnn(token, state, ln_out, blocks, head, emb):
    # print(state.shape)
    x = emb['weight'][token]
    w_ln0 = blocks[0]['ln0']
    x = layer_norm(x, **w_ln0)
    new_states = np.empty_like(state)
    for i in range(len(blocks)):
        block_w = blocks[i]

        xn_token = layer_norm(x, **blocks[i]['ln1'])
        r, k, v = token_mixing(xn_token, state[i, 1], **block_w['att'])

        state_wkv = state[i, 2:]
        # print(state_wkv.shape)

        x += rwkv(r, block_w['att']['output']['weight'], k, v, state_wkv, block_w['att']['time_first'])
        xn_channel = layer_norm(x, **blocks[i]['ln2'])
        xp = channel_mixing(xn_channel, state[i, 0], **block_w['ffn'])

        x += xp
        new_state = np.vstack([xn_channel,
                               xn_token,
                               *rwkv_state_flow(block_w['att']['time_decay'],
                                                state_wkv, k, v)])
        new_states = new_states.at[i].set(new_state)
    xn = layer_norm(x, **ln_out)
    logits = head['weight'] @ xn
    return logits, new_states


def rwkv_net_parallel(seq_len: int, tokens, states, ln_out, blocks, head, emb):
    """
    :param seq_len: int
    :param tokens: int32[seq_len]
    :param states: float32[n_blocks, 5, n_channels]
    :param ln_out:
    :param blocks:
    :param head:
    :param emb: {'weight': float32[n_vocab, n_channels]}
    :return:
    """
    assert seq_len >= 2
    rwkv_parallel = vmap(rwkv, in_axes=(0, None, 0, 0, 1, None), out_axes=0)
    empty_wkv_states = states[:, 2:, :]
    # assert empty_wkv_states.shape == (len(blocks), 3, seq_len, states.shape[2])
    prev_xn_token = states[:, 1, :]
    prev_xn_channel = states[:, 0, :]

    x = emb['weight'][tokens, :]
    w_ln0 = blocks[0]['ln0']
    x = layer_norm(x, **w_ln0)
    new_states = np.empty_like(states)
    states0 = np.empty_like(states)
    for i in range(len(blocks)):
        block_w = blocks[i]

        xn_token = layer_norm(x, **blocks[i]['ln1'])
        r, k, v = token_mixing_parallel(xn_token, prev_xn_token[i], **block_w['att'])

        state_wkv, _ = jax.lax.scan(
            partial(rwkv_state_flow_scan, block_w['att']['time_decay']),
            empty_wkv_states[0], (k, v),
        )
        print(state_wkv.shape)

        x += rwkv_parallel(r, block_w['att']['output']['weight'], k, v, state_wkv, block_w['att']['time_first'])
        xn_channel = layer_norm(x, **blocks[i]['ln2'])
        xp = channel_mixing_parallel(xn_channel, prev_xn_channel[i], **block_w['ffn'])

        x += xp
        new_states = new_states.at[i, :].set(np.stack([xn_channel[-2], xn_token[-2], *(s for s in state_wkv)]))
        states0 = states0.at[i, :].set(np.stack([xn_channel[0], xn_token[0], *(s for s in state_wkv)]))
    xn = layer_norm(x, **ln_out)
    # parallel version of `logits = head['weight'] @ xn`, t is time, c is channel, v is vocab
    logits = np.einsum('tc,vc->tv', xn, head['weight'])
    return logits, new_states, states0
# [Done] load params
# [Done] jit
# [TODO] associative scan
#  - [TODO] not trying scan, trying to formulate according to jianlin's blog (is cumsum in rwkv viable)?
# [TODO] pico-plus
# [TODO] training
