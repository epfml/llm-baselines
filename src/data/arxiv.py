import os
import tarfile
import logging
from pathlib import Path
from typing import Optional
from multiprocessing import Pool
from tempfile import NamedTemporaryFile
from subprocess import Popen, TimeoutExpired, PIPE

import numpy as np
import requests
from tqdm.auto import tqdm

from .tokenizing import encode

def convert_to_markdown(args: tuple[Path, Path]):
    texfile, mdroot = args
    mdfile = mdroot/f"{texfile.name}.md"
    with Popen(["pandoc", "--wrap=none", "--from", "latex", texfile,
                "--output", mdfile], stderr=PIPE) as proc:
        try:
            proc.communicate(timeout=1)
        except TimeoutExpired:
            proc.kill()



def fetch_arxiv(root: Path, year: int):
    # download latex
    url = f"https://www.cs.cornell.edu/projects/kddcup/download/hep-th-{year}.tar.gz"
    texroot = root/"tex"
    print("Downloading Arxiv year", year)
    req = requests.get(url, timeout=60)
    with NamedTemporaryFile(suffix=".tar.gz") as f:
        f.write(req.content)
        logging.debug("Tar saved in tempfile %s" % f.name)
        with tarfile.open(f.name) as tar:
            logging.debug("Extracting tarfile")
            tar.extractall(texroot)

    # convert to markdown
    mdroot = root/"md"/str(year)
    mdroot.mkdir(parents=True)
    files = list((texroot/str(year)).iterdir())
    with Pool(os.cpu_count()) as p:
        args = [(texfile, mdroot) for texfile in files]
        for _ in tqdm(p.imap_unordered(convert_to_markdown, args),
                      desc="Converting to markdown", total=len(files)):
            pass


def tokenize_arxiv(root: Path, tokenizer: str, year: int):
    tokens = []
    tokens_val = []
    tokens_test = []
    mds = root/"md"/str(year)

    # tokenize
    desc = f"Tokenizing {year}"
    for i, mdpath in enumerate(tqdm(list(mds.iterdir()), desc=desc)):
        with open(mdpath, encoding="utf8") as f:
            text = "".join(f.readlines())
        if i % 10 <= 6:  # train split
            tokens += encode(text, tokenizer)
        elif i % 10 <= 8:  # val split
            tokens_val += encode(text, tokenizer)
        else:  # test split
            tokens_test += encode(text, tokenizer)

    # save to dir
    tpath = root/tokenizer/str(year)
    tpath.mkdir(parents=True)
    for x, name in zip([tokens, tokens_val, tokens_test],
                       ["train", "val", "test"]):
        mem = np.memmap(tpath/f"{name}.npy", dtype=np.uint16, mode="w+",
                        shape=len(x))
        for i, v in enumerate(x):
            mem[i] = v


def load_arxiv(cachedir: Path, tokenizer: str, years: Optional[list[int]] = None):
    all_years = list(range(1992, 2004))
    if years is None:
        years = all_years
    assert set(years) <= set(all_years)
    root = cachedir/"arxiv"
    root.mkdir(exist_ok=True, parents=True)

    # download all years requested that are not present
    for year in years:
        if not (root/"md"/str(year)).exists():
            fetch_arxiv(root, year)

    # tokenize all years not previously tokenized
    for year in years:
        if not (root/tokenizer/str(year)).exists():
            tokenize_arxiv(root, tokenizer, year)

    # load meta
    ret = {}
    for split in ["train", "val"]:
        paths = [root/tokenizer/str(year)/f"{split}.npy" for year in years]
        x = [np.memmap(path, dtype=np.uint16, mode="r") for path in paths]
        ret[split] = np.concatenate(x)
    return ret


def get_arxiv_2000(tokenizer: str):
    return load_arxiv(Path(os.path.dirname(__file__))/"datasets", tokenizer, [2000])


def get_arxiv_full(tokenizer: str):
    return load_arxiv(Path(os.path.dirname(__file__))/"datasets", tokenizer)
