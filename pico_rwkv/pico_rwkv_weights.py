from typing import Callable, TypeVar, Iterable

from picojax.jax_utils import Arr, WeightsTree


def parse_rwkv_weight(keys: Iterable[str], get_tensor: Callable[[str], Arr], trim: bool=False) -> WeightsTree:
    w = {}
    for k in keys:
        parts = k.split('.')
        last = parts.pop()
        current_ = w
        for p in parts:
            if p.isdigit():
                p = int(p)
            if p not in current_:
                current_[p] = {}
            current_ = current_[p]
        current_[last] = get_tensor(k)

    for i in w['blocks'].keys():
        w['blocks'][i]['att']['kw'] = w['blocks'][i]['att']['key']['weight']
        w['blocks'][i]['att']['vw'] = w['blocks'][i]['att']['value']['weight']
        w['blocks'][i]['att']['rw'] = w['blocks'][i]['att']['receptance']['weight']
        w['blocks'][i]['ffn']['kw'] = w['blocks'][i]['ffn']['key']['weight']
        w['blocks'][i]['ffn']['vw'] = w['blocks'][i]['ffn']['value']['weight']
        w['blocks'][i]['ffn']['rw'] = w['blocks'][i]['ffn']['receptance']['weight']
        if trim:
            del w['blocks'][i]['att']['key']['weight']
            del w['blocks'][i]['att']['value']['weight']
            del w['blocks'][i]['att']['receptance']['weight']
            del w['blocks'][i]['ffn']['key']['weight']
            del w['blocks'][i]['ffn']['value']['weight']
            del w['blocks'][i]['ffn']['receptance']['weight']
    return w
