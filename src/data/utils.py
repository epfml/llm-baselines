import numpy as np
from typing import Dict
import torch

from .shakespeare import get_shakespeare_data
from .wikitext import get_wikitext_data
from .arxiv import get_arxiv_2000, get_arxiv_full
from .openwebtext2 import get_openwebtext2_data
from .slimpajama import get_slimpajama_data

import os
import requests

def get_dataset(args) -> Dict[str, np.ndarray]:
    """ Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
     contained in its own python file. The expected format at the moment is a dictionary of np.memmap
     containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data. """
    if args.dataset == 'wikitext':
        return get_wikitext_data()
    if args.dataset == "shakespeare-char":
        return get_shakespeare_data()
    if args.dataset == "shakespeare":
        return get_shakespeare() # return train_tokens, val_tokens
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

def token2seq(tokens, max_seq_length, seed=0):
    # return seq_shuffled, a list of seqs
    len_tokens = len(tokens)
    seq = []
    for i in range(len_tokens // max_seq_length):
        x = tokens[i*max_seq_length:(i+1)*max_seq_length]
        seq.append(x)
    
    return seq

def get_shakespeare(tokenizer):
    # return train_tokens, val_tokens, both are np.memmap
    char_tknzr = tokenizer.encode
    DATA_PATH = os.path.join(os.getcwd(), "datasets", "shakespeare")
    raw_path = os.path.join(DATA_PATH, "raw.txt")
    train_path = os.path.join(DATA_PATH, f"train.npy")
    test_path = os.path.join(DATA_PATH, f"test.npy")
    # if path is not even there, download all data
    if not os.path.exists(DATA_PATH):
        print("Downloading raw Shakespeare texts")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        os.makedirs(DATA_PATH, exist_ok=True)
        text = requests.get(url, timeout=60).text
        with open(raw_path, "w+", encoding="utf8") as f:
            f.write(text)
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Tokenizing Shakespeare texts")
        # load text
        with open(raw_path, encoding="utf8") as f:
            text = "".join(f.readlines())
        # encode text
        tokens = np.array(char_tknzr(text), dtype=np.uint16)
        # seqs, shuffled_id = token2seq(x_all, max_seq_length)
        # train = seqs[:int(0.8*len(seqs))]
        # val = seqs[int(0.8*len(seqs)):]
        # mem = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(len(x_seq_train), max_seq_length))
        # for i, x in enumerate(x_seq_train):
        #     mem[i] = x
        # mem = np.memmap(test_path, dtype=np.uint16, mode="w+", shape=(len(x_seq_test), max_seq_length))
        # for i, x in enumerate(x_seq_test):
        #     mem[i] = x
        train_tokens = tokens[:int(0.8*len(tokens))]
        val_tokens = tokens[int(0.8*len(tokens)):]
        # map memory
        mem = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=train_tokens.shape)
        mem[:] = train_tokens
        mem = np.memmap(test_path, dtype=np.uint16, mode="w+", shape=val_tokens.shape)
        mem[:] = val_tokens
    train_tokens = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_tokens = np.memmap(test_path, dtype=np.uint16, mode="r")
    print(f'Numer of tokens in Shakespeare, train: {len(train_tokens)}, val: {len(val_tokens)}')

    return train_tokens, val_tokens

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
        num_workers=0,
    )
    return loader, sampler


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        # chunk the data into sequences of length `sequence_length`
        # NOTE: we discard the last remainding sequence if it's not of length `sequence_length`
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.data[idx].astype(np.int64))
        return sample
    
def get_loader(dataset, batch_size, distributed_backend=None, seed=0):
    # dataset is a dict: {'train': a list of tokens with seq_len, 'val': eval_tokenized}
    train_dataset = MyDataset(dataset['train']) 
    val_dataset = MyDataset(dataset['val'])
    perm = np.random.RandomState(seed=seed).permutation(len(train_dataset))
    train_dataset = torch.utils.data.Subset(train_dataset, perm)
    print(f"dataset size (num seq: num_tokens / seq_len), train : {len(train_dataset)}, val: {len(val_dataset)}")

    if distributed_backend and distributed_backend.get_world_size() > 1:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.SequentialSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
    )
    if distributed_backend and distributed_backend.get_world_size() > 1:
        val_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=False)
    else:
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
    )
    print(f"num steps (num_tokens / seq_len / num_batch): {len(train_loader)}, val: {len(val_loader)}")
    if len(dataset['curated']) > 0:
        curated_dataset = MyDataset(dataset['curated'])
        curated_loader = torch.utils.data.DataLoader(
            curated_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        return {'train': train_loader, 'val': val_loader, 'curated': curated_loader, 'perm': perm, 'train_sampler': train_sampler}
    else:
        return {'train': train_loader, 'val': val_loader, 'perm': perm, 'train_sampler': train_sampler}