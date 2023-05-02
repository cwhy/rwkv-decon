from functools import partial

from jax import lax
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


def exp_mix_both(max_coef, k, v_s, v, base):
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

def exp_add(v1, v2, p1, p2):
    """
    Given
      x1 --> (v1, p1) --> v1 exp(p1),
      x2 --> (v2, p2) --> v2 exp(p2),
    calculate x1 + x2 and normalize it such that the
    largest exp exponent is 0,
      x1 + x2 --> (v1 exp(p1-p) + v2 exp(p2-p), p)
    where p = max(p1, p2)
    """
    p = np.maximum(p1, p2)
    return (v1 * np.exp(p1 - p) + v2 * np.exp(p2 - p)), p

def rwkv(state_wkv, r, k, v, ow, time_first, debug=False):
    """
    state_wkv: (v_state, wkv_bot, max_coef) / state[i, 2:].T
    """
    v_state, base_state, max_coef = state_wkv

    v_state, base_state, _ = exp_mix_both(max_coef, k + time_first, v_state, v, base_state)
    wkv = v_state / base_state
    return ow @ (r * wkv)


def rwkv_state_flow(time_decay, state_wkv, k, v, **kwargs):
    v_state, base_state, max_coef = state_wkv
    a, b, c = exp_mix_both(max_coef + time_decay, k, v_state, v, base_state)
    return a, b, c


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


def rwkv_scan(state, rs, ks, vs, ow, time_first, time_decay):
    w, u = time_decay, time_first

    def lru_scannable(carry, new):
        r, k, v = new
        new = rwkv(carry, r, k, v, ow, time_first)
        carry_new = rwkv_state_flow(w, carry, k, v)
        return np.stack(carry_new), new

    return lax.scan(lru_scannable, state, (rs, ks, vs))


def lru_parallel_scannable(left, right):
    (l_exp_kv, l_w), (r_exp_kv, r_w) = left, right
    return l_exp_kv * np.exp(r_w) + r_exp_kv, l_w + r_w


def rwkv_parallel_scan(seq_len: int, r, k, v, ow, time_first, time_decay):
    w, u = time_decay, time_first
    W = np.repeat(w[np.newaxis, :], seq_len, axis=0)

    exp_k = np.exp(k)
    v_state, _ = lax.associative_scan(lru_parallel_scannable, (exp_k * v, W))
    base_state, _ = lax.associative_scan(lru_parallel_scannable, (exp_k, W))
    curr_k = np.exp(u) * exp_k

    def shift1pad0(x):
        return np.pad(x, ((1, 0), (0, 0)), mode='constant', constant_values=0)[:-1, :]

    v_state = shift1pad0(v_state) + curr_k * v
    base_state = shift1pad0(base_state) + curr_k

    wkv = v_state / base_state
    return (r * wkv) @ ow.T


def rwkv_parallel_scan_alt(seq_len: int, r, k, v, ow, time_first, time_decay):
    w, u = time_decay, time_first
    W = np.repeat(w[np.newaxis, :], seq_len, axis=0)

    exp_k = np.exp(k)
    v_state, _ = lax.associative_scan(lru_parallel_scannable, (exp_k * v, W))
    base_state, _ = lax.associative_scan(lru_parallel_scannable, (exp_k, W))
    curr_diff = exp_k * (np.exp(u + w) - 1)

    v_state += curr_diff * v
    base_state += curr_diff
    base_state += 10e-6

    wkv = v_state / base_state
    return (r * wkv) @ ow.T

def lru_parallel_scannable_normalized(left, right):
    (l_exp_kv, l_w, p_w), (r_exp_kv, r_w, p_r) = left, right
    p = np.maximum(p_w + r_w, p_r)
    return l_exp_kv * np.exp(r_w + p_w - p) + r_exp_kv * np.exp(p_r - p), l_w + r_w, p


def rwkv_parallel_scan_stable(r, k, v, ow, time_first, time_decay):
    w, u = time_decay, time_first
    W = np.repeat(w[np.newaxis, :], v.shape[0], axis=0)
    ones = np.ones_like(k)

    a_state, _, p_state = lax.associative_scan(lru_parallel_scannable_normalized, (v, W, k))
    b_state, _, _ = lax.associative_scan(lru_parallel_scannable_normalized, (ones, W, k))

    c, _ = exp_add(a_state, v, p_state, u+w+k)
    d, _ = exp_add(b_state, ones, p_state, u+w+k)

    wkv = c / d
    return (r * wkv) @ ow.T


# @jit
def rwkv_net_rnn(token, state, ln_out, blocks, head, emb):
    # print(state.shape)
    # print(token)
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

        x += rwkv(state_wkv, r, k, v, block_w['att']['output']['weight'], block_w['att']['time_first'], debug=False)
        xn_channel = layer_norm(x, **blocks[i]['ln2'])
        xp = channel_mixing(xn_channel, state[i, 0], **block_w['ffn'])

        x += xp
        new_state = np.vstack([xn_channel,
                               xn_token,
                               *rwkv_state_flow(block_w['att']['time_decay'],
                                                state_wkv, k, v, debug=False)])
        new_states = new_states.at[i].set(new_state)
    xn = layer_norm(x, **ln_out)
    logits = head['weight'] @ xn
    return logits, new_states


def rwkv_net_scan(seq_len: int, tokens, states, ln_out, blocks, head, emb):
    assert seq_len >= 2
    prev_xn_token = states[:, 1, :]
    prev_xn_channel = states[:, 0, :]
    prev_wkv_states = states[:, 2:, :]

    x = emb['weight'][tokens, :]
    w_ln0 = blocks[0]['ln0']
    x = layer_norm(x, **w_ln0)
    new_states = np.empty_like(states)
    for i in range(len(blocks)):
        block_w = blocks[i]

        xn_token = layer_norm(x, **blocks[i]['ln1'])
        r, k, v = token_mixing_parallel(xn_token, prev_xn_token[i], **block_w['att'])

        state_wkv, xp = rwkv_scan(prev_wkv_states[i], r, k, v, block_w['att']['output']['weight'],
                                  block_w['att']['time_first'], block_w['att']['time_decay'])
        x += xp
        xn_channel = layer_norm(x, **blocks[i]['ln2'])
        xp = channel_mixing_parallel(xn_channel, prev_xn_channel[i], **block_w['ffn'])

        x += xp
        new_states = new_states.at[i, :].set(np.stack([xn_channel[-1], xn_token[-1], *(s for s in state_wkv)]))
    xn = layer_norm(x, **ln_out)
    # parallel version of `logits = head['weight'] @ xn`, t is time, c is channel, v is vocab
    logits = np.einsum('tc,vc->tv', xn, head['weight'])
    return logits, new_states


@partial(jit, static_argnums=(0,))
def rwkv_net_parallel(seq_len: int, tokens, ln_out, blocks, head, emb):
    """
    :param seq_len: int
    :param tokens: int32[seq_len]
    :param ln_out:
    :param blocks:
    :param head:
    :param emb: {'weight': float32[n_vocab, n_channels]}
    :return:
    """
    assert seq_len >= 2
    zeros_padding = np.zeros_like(emb['weight'][0, :])

    x = emb['weight'][tokens, :]
    w_ln0 = blocks[0]['ln0']
    x = layer_norm(x, **w_ln0)
    for i in range(len(blocks)):
        block_w = blocks[i]

        xn_token = layer_norm(x, **blocks[i]['ln1'])
        r, k, v = token_mixing_parallel(xn_token, zeros_padding, **block_w['att'])

        xp = rwkv_parallel_scan_stable(r, k, v, block_w['att']['output']['weight'],
                                block_w['att']['time_first'], block_w['att']['time_decay'])
        x += xp
        xn_channel = layer_norm(x, **blocks[i]['ln2'])
        xp = channel_mixing_parallel(xn_channel, zeros_padding, **block_w['ffn'])

        x += xp
    xn = layer_norm(x, **ln_out)
    # parallel version of `logits = head['weight'] @ xn`, t is time, c is channel, v is vocab
    logits = np.einsum('tc,vc->tv', xn, head['weight'])
    return logits

# [Done] load params
# [Done] jit
# [TODO] associative scan
#  - trying to formulate according to jianlin's blog (is cumsum in rwkv viable)?
#  - not trying to use states, only use it for training and fixed-context inference
#  - [Done] make it work
#  - [Done] check numerical issues with help of yilun
# [Done] make a scan version and compare with
#  - [Done] scan batch of tokens
# - [TODO] maintain state using normalized version of scan
# [TODO] pico-plus
# [TODO] training
