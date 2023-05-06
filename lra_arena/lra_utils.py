from pathlib import Path
from typing import NamedTuple, Sequence, Callable, Generator, Iterator

import jax.numpy as np
from jax import random
from s5.dataloading import Datasets, DataLoader

from picojax.random_utils import SafeKey
from picojax.train_utils import MixedLenBatchType


class LRABatchConfig(NamedTuple):
    block_size: int
    batch_size: int
    s5_dataloaders: DataLoader
    train_size: int
    n_classes_in: int
    n_classes_out: int

    @classmethod
    def from_s5(cls, batch_size: int, cache_path: Path, dataset_name: str, seed: int = 0):
        create_dataset_fn = Datasets[dataset_name]
        trainloader, valloader, testloader, aux_dataloaders, n_classes, seq_len, in_dim, train_size = create_dataset_fn(
            cache_path, seed=seed, bsz=batch_size)
        return cls(block_size=seq_len, batch_size=batch_size,
                   s5_dataloaders={'train': trainloader, 'val': valloader, 'test': testloader}, train_size=train_size,
                     n_classes_in=in_dim, n_classes_out=n_classes)

    @property
    def samplers(self) -> dict[str, Callable[[SafeKey], MixedLenBatchType]]:
        def get_sampler(loader: DataLoader) -> Callable[[SafeKey], MixedLenBatchType]:
            loader_sampler = iter(loader.batch_sampler)
            def sampler(key: SafeKey):
                x, y, l = next(loader_sampler)
                return np.array(x), np.array(y), np.array(l['lengths'])

            return sampler

        return {k: get_sampler(v) for k, v in self.s5_dataloaders.items()}

    @property
    def dataloaders(self) -> dict[str, Iterator[MixedLenBatchType]]:
        def get_dataloader(loader: DataLoader) -> Iterator[MixedLenBatchType]:
            def data_generator() -> Iterator[MixedLenBatchType]:
                loader_iter = iter(loader)
                while True:
                    x, y, l = next(loader_iter)
                    yield np.array(x), np.array(y), np.array(l['lengths'])

            return data_generator()
        return {k: get_dataloader(v) for k, v in self.s5_dataloaders.items()}
