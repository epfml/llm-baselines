import numpy as np

from .shakespeare import get_shakespeare_data
from .wikitext import get_wikitext_data
from .arxiv import get_arxiv_2000, get_arxiv_full


def get_dataset(args) -> dict[str, np.ndarray]:
    """ Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
     contained in its own python file. The expected format at the moment is a dictionary of np.memmap
     containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data. """
    if args.dataset == 'wikitext':
        return get_wikitext_data()
    if args.dataset == "shakespeare":
        return get_shakespeare_data(args.tokenizer)
    if args.dataset == "arxiv2000":
        return get_arxiv_2000(args.tokenizer)
    if args.dataset == "arxiv":
        return get_arxiv_full(args.tokenizer)
    else:
        raise NotImplementedError(f"Unknow dataset key '{args.dataset}'")
