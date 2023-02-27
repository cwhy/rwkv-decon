# %%
import jax
import jax.numpy as jnp

"""
mk o i g u K R V mv mv
fk fi fK fR fv
ln0b ln0w ln1b ln1w ln2b ln2w
"""

sigma = jax.nn.sigmoid
maxi = jax.numpy.maximum
e = jax.numpy.exp
relu = jax.nn.relu


def mix(_in, _state, _mix):
    return _in * _mix + _state * (1 - _mix)


"""
x: 1024
s: 1024

"""


def ln(x, w, b):
    mean = jnp.mean(x)
    v = jnp.var(x)
    o = x - mean
    i = w * jax.lax.rsqrt(v + 1e-5)
    return o * i + b


def mix_t(x, s, K, V, R, mk, o, mr, g, u, mv, aa, bb, pp):
    xk, xv, xr = mix(x, s, mk), mix(x, s, mv), mix(x, s, mr)
    s.update_(x)

    k = xk @ K
    v = xv @ V

    ww = u + k
    p = maxi(pp, ww)
    wkv = (e(pp - p) * aa + e(ww - p) * v) / (e(pp - p) * bb + e(ww - p))

    p = maxi(pp + g, k)
    aa.update_(e(ww - p) * aa + e(k - p) * v)
    bb.update_(e(ww - p) * bb + e(k - p))
    pp.update_(p)

    return (sigma(xr @ R) * wkv) @ o


def mix_h(x, ss, K, R, mk, mr, V):
    xk, xr = mix(x, ss, mk), mix(x, ss, mr)
    ss.update_(x)
    return sigma(xr @ R) * (relu(xk @ K) ** 2 @ V)


def cell(x, s, K, V, R, mk, o, mr, g, u, mv, aa, bb, pp, ln1b, ln1w, ln2b, ln2w, fK, fR, mfk, mfr, fV):
    xx = ln(x, ln1w, ln1b)
    x += mix_t(xx, s, K, V, R, mk, o, mr, g, u, mv, aa, bb, pp)
    x = ln(x, ln2w, ln2b)
    xx = mix_h(x, s, fK, fR, mfk, mfr, fV)
    return x + xx
