import jax.numpy as np
from jax import lax

from pico_rwkv.pico_rwkv_parallel import token_mixing_parallel, channel_mixing_parallel
from pico_rwkv.pico_rwkv_rnn import layer_norm, rwkv, rwkv_state_flow


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
