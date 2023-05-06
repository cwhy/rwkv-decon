from pathlib import Path
from s5.dataloading import Datasets
from typing import NamedTuple, Iterable, Union, Collection, Sequence, Callable

from jax import random

from picojax.jax_utils import Arr
from picojax.random_utils import SafeKey
from picojax.train_utils import BatchType


class LRABatchConfig(NamedTuple):
    block_size: int
    batch_size: int
    dataloaders: dict[str, Sequence[BatchType]]
    train_size: int
    n_classes_in: int
    n_classes_out: int

    @classmethod
    def from_s5(cls, batch_size: int, cache_path: Path, dataset_name: str, seed: int = 0):
        create_dataset_fn = Datasets[dataset_name]
        trainloader, valloader, testloader, aux_dataloaders, n_classes, seq_len, in_dim, train_size = create_dataset_fn(
            cache_path, seed=seed, bsz=batch_size)
        return cls(block_size=seq_len, batch_size=batch_size,
                   dataloaders={'train': trainloader, 'val': valloader, 'test': testloader}, train_size=train_size,
                     n_classes_in=in_dim, n_classes_out=n_classes)

    @property
    def samplers(self) -> dict[str, Callable[[SafeKey], BatchType]]:
        def get_sampler(loader: Sequence[BatchType]) -> Callable[[SafeKey], BatchType]:
            def sampler(key: SafeKey):
                idx = random.randint(key.get(), (), 0, self.train_size).item()
                return loader[idx]

            return sampler

        return {k: get_sampler(v) for k, v in self.dataloaders.items()}
