import numpy as np
from typing import Dict
import torch

from .shakespeare import get_shakespeare_data
from .wikitext import get_wikitext_data
from .openwebtext2 import get_openwebtext2_data
from .slimpajama import get_slimpajama_data


def get_dataset(args) -> Dict[str, np.ndarray]:
    """Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
    contained in its own python file. The expected format at the moment is a dictionary of np.memmap
    containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data.
    This just returns a dictionary of the paths to the np.memmap objects, and does not load the data into memory.
    """
    if args.dataset == "wikitext":
        return get_wikitext_data()
    if args.dataset == "shakespeare-char":
        return get_shakespeare_data()
    if args.dataset == "openwebtext2":
        return get_openwebtext2_data()
    if args.dataset == "slimpajama":
        return get_slimpajama_data()
    else:
        raise NotImplementedError(f"Unknow dataset key '{args.dataset}'")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, sequence_length):
        super().__init__()
        self.data_path = data_path
        self.sequence_length = sequence_length

    def __len__(self):
        data = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        total_length = len(data)
        # chunk the data into sequences of length `sequence_length`
        # NOTE: we discard the last remaining sequence if it's not of length `sequence_length`
        return (total_length - 1) // self.sequence_length

    def __getitem__(self, idx):
        data = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        seq_length = self.sequence_length
        idx = idx * seq_length
        x = torch.from_numpy((data[idx : idx + seq_length]).astype(np.int64))
        y = torch.from_numpy((data[idx + 1 : idx + 1 + seq_length]).astype(np.int64))
        return x, y


def get_dataloader(data, sequence_length, batch_size, seed=0, distributed_backend=None):
    """Create a DataLoader for the given data. If distributed_backend is provided and is truly
    distributed (world size > 1), the DataLoader will be created with a DistributedSampler that
    splits the data across the processes (in conjunction with DDP).
    Otherwise, use a RandomSampler with the specified seed.

    Returns both the dataloader and the sampler.
    """
    dataset = Dataset(data, sequence_length=sequence_length)
    if distributed_backend and distributed_backend.get_world_size() > 1:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            shuffle=True,
            seed=seed,
        )
    else:
        g = torch.Generator()
        g.manual_seed(seed)
        sampler = torch.utils.data.RandomSampler(
            dataset, replacement=False, generator=g
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=4,
    )
    return loader, sampler
