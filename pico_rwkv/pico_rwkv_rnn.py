import jax.numpy as np
from jax import lax, jit
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


@jit
def rwkv_net_rnn(token, state, ln_out, blocks, head, emb):
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
