from functools import partial

import jax.numpy as np
from jax import jit, vmap, lax

from pico_rwkv.pico_rwkv_rnn import channel_mixing, token_mixing, layer_norm


def exp_mix(v1, v2, p1, p2):
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


def token_mixing_parallel(x, x_prev1, time_mix_k, time_mix_v, time_mix_r, kw, vw, rw, **kwargs):
    x_prev = np.vstack((x_prev1, x[:-1, :]))
    token_mixing_p = vmap(token_mixing, in_axes=(0, 0, None, None, None, None, None, None), out_axes=(0, 0, 0))
    return token_mixing_p(x, x_prev, time_mix_k, time_mix_v, time_mix_r, kw, vw, rw)


def channel_mixing_parallel(x, x_prev1, time_mix_k, time_mix_r, kw, vw, rw, **kwargs):
    x_prev = np.vstack((x_prev1, x[:-1, :]))
    channel_mixing_p = vmap(channel_mixing, in_axes=(0, 0, None, None, None, None, None), out_axes=0)
    return channel_mixing_p(x, x_prev, time_mix_k, time_mix_r, kw, vw, rw)


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

    c, _ = exp_mix(a_state, v, p_state, u + w + k)
    d, _ = exp_mix(b_state, ones, p_state, u + w + k)

    wkv = c / d
    return (r * wkv) @ ow.T


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
# - [TODO] merge upper and lower scan
# [Abandoned] pico-plus
# [Done] training
