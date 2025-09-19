<<<<<<< HEAD
import numpy as np
from typing import Dict
import torch

from .shakespeare import get_shakespeare_data
from .wikitext import get_wikitext_data
from .arxiv import get_arxiv_2000, get_arxiv_full
from .openwebtext2 import get_openwebtext2_data
from .slimpajama import get_slimpajama_data
from .benchmarks import get_mathqa


def get_dataset(args) -> Dict[str, np.ndarray]:
    """ Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
     contained in its own python file. The expected format at the moment is a dictionary of np.memmap
     containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data. """
    if args.dataset == 'wikitext':
        return get_wikitext_data()
    if args.dataset == "shakespeare-char":
        return get_shakespeare_data()
    if args.dataset == "arxiv2000":
        return get_arxiv_2000()
    if args.dataset == "arxiv":
        return get_arxiv_full()
    if args.dataset == "arxiv+wiki":
        arxiv_data = get_arxiv_full()
        wiki_data = get_wikitext_data()
        train_data = np.concatenate((arxiv_data['train'], wiki_data['train']))
        val_data = np.concatenate((arxiv_data['val'], wiki_data['val']))
        return {'train': train_data, 'val': val_data}
    if args.dataset == 'openwebtext2':
        return get_openwebtext2_data()
    if args.dataset == "slimpajama":
        return get_slimpajama_data()
    if args.dataset == "mathqa":
        return get_mathqa()
    else:
        raise NotImplementedError(f"Unknow dataset key '{args.dataset}'")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, sequence_length):
        super().__init__()
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        total_length = len(self.data)
        # chunk the data into sequences of length `sequence_length`
        # NOTE: we discard the last remainding sequence if it's not of length `sequence_length`
        return (total_length - 1) // self.sequence_length

    def __getitem__(self, idx):
        seq_length = self.sequence_length
        idx = idx * seq_length
        x = torch.from_numpy((self.data[idx : idx + seq_length]).astype(np.int64))

        y = torch.from_numpy(
            (self.data[idx + 1 : idx + 1 + seq_length]).astype(np.int64)
        )
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
=======
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist

from .arxiv import get_arxiv_2000, get_arxiv_full
from .c4 import get_c4_data
from .fineweb import get_fineweb_data
from .fineweb_edu import get_fineweb_edu_data
from .openwebtext2 import get_openwebtext2_data
from .redpajama import get_redpajama_data, get_redpajamav2_data
from .shakespeare import get_shakespeare_data
from .slimpajama import get_slimpajama_data
from .wikitext import get_wikitext_data


def get_dataset(args) -> Dict[str, np.ndarray]:
    """Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
    contained in its own python file. The expected format at the moment is a dictionary of np.memmap
    containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data.
    """
    if args.dataset == "wikitext":
        return get_wikitext_data(args.datasets_dir)
    if args.dataset == "shakespeare-char":
        return get_shakespeare_data(args.datasets_dir)
    if args.dataset == "arxiv2000":
        return get_arxiv_2000(args.datasets_dir)
    if args.dataset == "arxiv":
        return get_arxiv_full(args.datasets_dir)
    if args.dataset == "arxiv+wiki":
        arxiv_data = get_arxiv_full(args.datasets_dir)
        wiki_data = get_wikitext_data(args.datasets_dir)
        train_data = np.concatenate((arxiv_data["train"], wiki_data["train"]))
        val_data = np.concatenate((arxiv_data["val"], wiki_data["val"]))
        return {"train": train_data, "val": val_data}
    if args.dataset == "openwebtext2":
        return get_openwebtext2_data(args.datasets_dir)
    if args.dataset == "redpajama":
        return get_redpajama_data(args.datasets_dir)
    if args.dataset == "redpajamav2":
        return get_redpajamav2_data(args.datasets_dir)
    if args.dataset == "slimpajama":
        return get_slimpajama_data(args.datasets_dir)
    if args.dataset == "fineweb":
        return get_fineweb_data(args.datasets_dir)
    if args.dataset == "finewebedu":
        return get_fineweb_edu_data(args.datasets_dir)
    if args.dataset == "c4":
        return get_c4_data(args.datasets_dir)
    else:
        raise NotImplementedError(f"Unknow dataset key '{args.dataset}'")


class DataReader:
    def __init__(
        self,
        data_src,
        batch_size,
        sequence_length,
        seed=1337,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=False,
    ):
        if isinstance(data_src, (str, Path)):
            self.data_path = Path(data_src)
            self.keep_in_ram = keep_in_ram
            if keep_in_ram:
                self.data = np.array(
                    np.memmap(self.data_path, dtype=np.uint16, mode="r")
                )
            else:
                self.data = None
        elif isinstance(data_src, (np.ndarray, np.memmap)):
            self.data_path = None
            self.data = data_src
            self.keep_in_ram = True

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.seed = seed
        self.with_replacement = with_replacement

        self.num_tokens = len(self._get_data())

        if auto_shard and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            print(
                f"Distributed DataReader Initialized for Worker {self.rank}/{self.world_size}"
            )
        else:
            self.world_size = 1
            self.rank = 0

        # Sampling without replacement
        self.last_epoch = None
        self.order = None
        self.epoch_offset = None
        self.step = 0
        self.num_batches_of_seqlen = 0
        if not with_replacement:
            self._shuffle_epoch(0)

    def __len__(self):
        # Length in valid start indices for a sequence
        # Extra -1 to have a valid next token for the final token of the last idx
        return self.num_tokens - self.sequence_length - 1

    def _get_data(self):
        if self.data is not None:
            return self.data
        else:
            # Construct the memmap each time to avoid a memory leak per NanoGPT
            # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
            return np.memmap(self.data_path, dtype=np.uint16, mode="r")

    def __getitem__(self, idx):
        # Return the underlying datapoint, no random sampling, no worker sharding
        assert 0 <= idx < len(self)
        data = self._get_data()
        x = torch.from_numpy(data[idx : idx + self.sequence_length].astype(np.int64))
        y = torch.from_numpy(
            data[idx + 1 : idx + self.sequence_length + 1].astype(torch.int64)
        )
        return x, y

    def set_step(self, step):
        self.step = step

    def sample_batch(self):
        data = self._get_data()

        if self.with_replacement:
            idxs = self._sample_with_replacement(self.step)
        else:
            idxs = self._sample_without_replacement(self.step)
        self.step += 1

        xy = np.stack([data[i : i + self.sequence_length + 1] for i in idxs]).astype(
            np.int64
        )
        x = torch.from_numpy(xy[:, :-1]).contiguous()
        y = torch.from_numpy(xy[:, 1:]).contiguous()
        return x, y

    def _sample_with_replacement(self, idx):
        # Return an array of token indices of length self.batch_size
        # Sampled with replacement, can get repeats at any time
        seed = self.seed + idx * self.world_size + self.rank
        rng = np.random.default_rng(seed)
        return rng.integers(len(self), self.batch_size)

    def _shuffle_epoch(self, epoch):
        seed = self.seed + epoch
        rng = np.random.default_rng(seed)
        # Drop one sequence to allow different offsets per epoch:
        self.order = rng.permutation((len(self)) // self.sequence_length - 1)
        # Shift all sequences in this epoch by this amount:
        self.epoch_offset = rng.integers(self.sequence_length)
        self.last_epoch = epoch
        self.num_batches_of_seqlen = (
            len(self.order) // self.batch_size
        )  # Drops remainder batch

    def _sample_without_replacement(self, step):
        # Return an array of token indices of length self.batch_size
        # Sampled without replacement, cycle all sequences before potential repeats
        # Sequences are randomly offset in every epoch as well
        batch_idx = self.world_size * step + self.rank
        epoch_length = self.num_batches_of_seqlen

        epoch = batch_idx // epoch_length
        if epoch != self.last_epoch:
            self._shuffle_epoch(epoch)
        epoch_idx = batch_idx % epoch_length

        start = epoch_idx * self.batch_size
        end = start + self.batch_size
        return self.order[start:end] * self.sequence_length + self.epoch_offset

    def num_batches(self):
        if self.with_replacement:
            return self.num_tokens // self.batch_size
        return self.num_batches_of_seqlen
>>>>>>> otherfork/main
