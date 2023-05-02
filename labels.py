from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Union


@dataclass(frozen=True)
class Labels:
    tags: frozenset[str]

    def fmt(self) -> str:
        return ', '.join(self.tags)

    def check(self) -> None:
        if not isinstance(self.tags, frozenset):
            raise ValueError(f"tags must be a frozenset, not {type(self.tags)}")

    def covers(self, other: Labels) -> bool:
        return self.tags >= other.tags

    def __contains__(self, item: str) -> bool:
        return item in self.tags

    def __len__(self) -> int:
        return len(self.tags)

    def __add__(self, other: Union[str, Labels]) -> Labels:
        if isinstance(other, str):
            return Labels(tags=self.tags | {other})
        else:
            return Labels(tags=self.tags | other.tags)

    def __radd__(self, other):
        return self + other

    @classmethod
    def empty(cls):
        return cls(frozenset())

    @classmethod
    def from_strs(cls, *tags: str) -> Labels:
        return cls(frozenset(tags))



L = Labels.from_strs


@runtime_checkable
class WithLabels(Protocol):
    labels: Labels

    def clear_labels(self) -> WithLabels:
        ...
