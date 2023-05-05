# to download data: https://github.com/google-research/long-range-arena
# https://storage.googleapis.com/long-range-arena/lra_release.gz unzip
from pathlib import Path

# load data in /Data/LRA/lra_release/lra_release/
# clone https://github.com/lindermanlab/S5/tree/main repo and `pip install -e .` in the repo
from s5.dataloading import Datasets
print(Datasets.keys())
create_dataset_fn = Datasets['listops-classification']
batch_size = 8
seed = 0
data_path = Path("/Data")
trainloader, valloader, testloader, aux_dataloaders, n_classes, seq_len, in_dim, train_size = create_dataset_fn(data_path, seed=0, bsz=batch_size)