from dataclasses import dataclass
from functools import reduce
from typing import NamedTuple, Callable, Sequence, Protocol

import sympy
from sympy import symbols, init_printing, Eq, sin, cos, Idx, IndexedBase, pprint, Function, Piecewise, Tuple, Mul, Add, \
    lambdify, simplify, expand, Sum, Indexed, Expr, exp

a, b, c, d, l = symbols('a, b, c, d, λ')

t = symbols('t', integer=True)
i = Idx('i', t)
j = Idx('j', t)
k = Idx('k', t)

Z = IndexedBase('z', shape=(t,))
Y = IndexedBase('y', shape=(t,))
L = IndexedBase('λ', shape=(t,))

init_printing(use_unicode=True)

gen_next = Eq(Y[t + 1], Y[t] * l + Z[t])
pprint(gen_next, use_unicode=True)


def is_associative(op, a, b, c) -> bool:
    return op(op(a, b), c).equals(op(a, op(b, c)))


def is_distributive(mul_like, add_like, a, b, c) -> bool:
    return mul_like(a, add_like(b, c)).equals(add_like(mul_like(a, b), mul_like(a, c)))

class IndexedBaseG(Protocol):
    def __getitem__(self, item) -> Expr:
        ...


class RecPair(NamedTuple):
    weight: Expr
    bias: Expr

    def equals(self, other: 'RecPair') -> bool:
        return self.weight.equals(other.weight) and self.bias.equals(other.bias)


class IndexedPair(NamedTuple):
    weight: IndexedBaseG
    bias: IndexedBaseG

    def __getitem__(self, item) -> RecPair:
        return RecPair(self.weight[item], self.bias[item])


class RecFormula(NamedTuple):
    x: IndexedBase
    params: IndexedPair
    op_mul: Function = Mul
    op_add: Function = Add

    def test(self):
        assert is_associative(self.op_mul, a, b, c)
        assert is_associative(self.op_add, a, b, c)
        assert is_distributive(self.op_mul, self.op_add, a, b, c)

    @property
    def rec_base(self) -> Indexed:
        return self.params[0].bias

    def rec_step(self, n: int):
        base_ = self.rec_base
        for i in range(1, n):
            w, b = self.params[i]
            base_ = self.op_add(self.op_mul(base_, w), b)
        return base_

    def pscan_i(self, i: int) -> RecPair:
        return self.params[i]

    @property
    def pscan_base(self) -> RecPair:
        return self.pscan_i(0)

    def pscan_step(self, left: RecPair, right: RecPair) -> RecPair:
        self.test()
        return RecPair(simplify(self.op_mul(left.weight, right.weight)),
                       simplify(expand((self.op_add(self.op_mul(left.bias, right.weight),
                                   right.bias)))))


@dataclass
class IndexedExpr:
    base_vars: list[IndexedBaseG]
    expr: Callable[[Sequence[Expr]], Expr]

    def __getitem__(self, item) -> Expr:
        return self.expr([v[item] for v in self.base_vars])


@dataclass
class FakeIndex:
    const: Expr

    def __getitem__(self, item) -> Indexed:
        return self.const

# print(build_rec_formula(Y, t, Z, L))
f = RecFormula(Y, IndexedPair(FakeIndex(l), Z))
# pprint(f.rec_equation, use_unicode=True)
pprint(f.pscan_i(i), use_unicode=True)
pprint(f.pscan_base, use_unicode=True)
ne = f.pscan_step(f.pscan_base, f.pscan_i(1))
pprint(ne, use_unicode=True)
pprint(f.pscan_step(ne, f.pscan_i(2)), use_unicode=True)
assert is_associative(Mul, a, b, c)
assert is_associative(f.pscan_step, f.pscan_i(i), f.pscan_i(j), f.pscan_i(k))

base_ = f.pscan_base
for i in range(1, 10):
    base_ = f.pscan_step(base_, f.pscan_i(i))
pprint(base_, use_unicode=True)
rec10 = f.rec_step(10)
pprint(rec10, use_unicode=True)
assert base_[1].equals(f.rec_step(10))
print(simplify(expand(rec10)))
print(simplify(expand(base_[1])))


#%%
K = IndexedBase('k', shape=(t,))
V = IndexedBase('v', shape=(t,))
w = symbols('w')

lc = exp(w)
lci = FakeIndex(lc)
expKV = IndexedExpr([K, V], lambda kv: exp(kv[0]) * kv[1])
f = RecFormula(Y, IndexedPair(lci, expKV), op_mul=Mul, op_add=Add)
rec10 = f.rec_step(3)
print(rec10)
base_ = f.pscan_base
for i in range(1, 3):
    base_ = f.pscan_step(base_, f.pscan_i(i))
print(base_[1])
print(base_[0])
assert base_[1].equals(f.rec_step(3))
print(simplify(expand(f.rec_step(5))))

i = Idx('i', t)
dummy_l = RecPair(lc, expKV[i])
dummy_r = RecPair(lc, expKV[i + 1])
ne = f.pscan_step(dummy_l, dummy_r)
print(ne[0])
print(ne[1])

# This can be alternative associative scan function of rwkv
# version 2
def pscan_step_alt(left: RecPair, right: RecPair) -> RecPair:
    return RecPair(simplify(left.weight + right.weight),
                   simplify(expand((left.bias * exp(right.weight) + right.bias))))

#i = Idx('i', t)
res = reduce(pscan_step_alt, [RecPair(w, expKV[j]) for j in range(1, 5)])
print(res)
res_rev = reduce(lambda a, b: pscan_step_alt(b, a), [RecPair(w, expKV[j]) for j in reversed(range(1, 5))])
print(res_rev)
print(simplify(expand(f.rec_step(5))))

