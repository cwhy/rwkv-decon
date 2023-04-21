import jax.numpy as np
from jax.lax import rsqrt
from jax.nn import sigmoid, relu


def layer_norm_org(x, g, b, eps: float = 1e-5):
    print(x.shape)
    print(g.shape)
    print(b.shape)
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) * rsqrt(variance + eps) + b


def projection(x, state, i: int, time_mix_k, time_mix_v, time_mix_r, kw, vw, rw):
    xk = x * time_mix_k + state[5 * i + 1] * (1 - time_mix_k)
    xv = x * time_mix_v + state[5 * i + 1] * (1 - time_mix_v)
    xr = x * time_mix_r + state[5 * i + 1] * (1 - time_mix_r)

    state = state.at[5 * i + 1].set(x)

    r = sigmoid(rw @ xr)
    k = kw @ xk
    v = vw @ xv
    return r, k, v, state


def rwkv(r, ow, k, v, state, i: int, time_first, time_decay):
    # state : [50, 1024]
    # r, k, v, time_first, aa, bb, pp : [1024, ]
    aa = state[5 * i + 2]
    bb = state[5 * i + 3]
    pp = state[5 * i + 4]
    ww = time_first + k
    qq = np.maximum(pp, ww)
    e1 = np.exp(pp - qq)
    e2 = np.exp(ww - qq)
    a = e1 * aa + e2 * v
    b = e1 * bb + e2
    wkv = a / b

    ww = pp + time_decay
    qq = np.maximum(ww, k)
    e1 = np.exp(ww - qq)
    e2 = np.exp(k - qq)
    state = state.at[5 * i + 2].set(e1 * aa + e2 * v)
    state = state.at[5 * i + 3].set(e1 * bb + e2)
    state = state.at[5 * i + 4].set(qq)
    return ow @ (r * wkv), state


def time_mixing(x, state, i: int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
    r, k, v, state = projection(x, state, i, time_mix_k, time_mix_v, time_mix_r, kw, vw, rw)
    return rwkv(r, ow, k, v, state, i, time_first, time_decay)


def channel_mixing(x, state, i: int, time_mix_k, time_mix_r, kw, vw, rw):
    xk = x * time_mix_k + state[5 * i + 0] * (1 - time_mix_k)
    xr = x * time_mix_r + state[5 * i + 0] * (1 - time_mix_r)
    state = state.at[5 * i + 0].set(x)
    r = sigmoid(rw @ xr)
    k = np.square(relu(kw @ xk))  # square relu, primer paper
    return r * (vw @ k), state


def block(x, state, i: int, att, ffn, ln1, ln2):
    xn = layer_norm(x, ln1.weight, ln1.bias)
    xp, state = time_mixing(xn, state, i, att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_first,
                            att.time_decay,
                            att.key.weight, att.value.weight, att.receptance.weight, att.output.weight)
    x += xp
    xn = layer_norm(x, ln2.weight, ln2.bias)
    xp, state = channel_mixing(xn, state, i, ffn.time_mix_k, ffn.time_mix_r, ffn.key.weight, ffn.value.weight,
                               ffn.receptance.weight)
    print(xp)
    raise Exception("stop")
    x += xp
    return x, state


def rwkv_net(token, state, ln_out, blocks, head, emb):
    x = emb.weight[token]
    x = layer_norm(x, blocks[0].ln0.weight, blocks[0].ln0.bias)
    for i in range(len(blocks)):
        block_w = blocks[i]
        x, state = block(x, state, i, block_w.att, block_w.ffn, block_w.ln1, block_w.ln2)
    xn = layer_norm(x, ln_out.weight, ln_out.bias)
    x = head.weight @ xn
    return x, state


def rwkv_net_w(token, state, w):
    return rwkv_net(token, state, w.ln_out, w.blocks, w.head, w.emb)

# load params
# associative scan
# pico-plus
