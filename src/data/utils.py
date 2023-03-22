import numpy as np
import torch

from .wikitext import get_wikitext_data


def get_dataset(args):
    """ Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
     contained in its own pythin file. The expected format at the moment is a disctionary of np.memmap
     containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data. """
    if args.dataset == 'wikitext':
        return get_wikitext_data()
    else:
        raise NotImplementedError(f"Unknow dataset key '{args.dataset}'")