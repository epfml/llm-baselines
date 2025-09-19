<<<<<<< HEAD
import os
import tarfile
import logging
from pathlib import Path
from typing import Optional
from multiprocessing import Pool
from tempfile import NamedTemporaryFile
from subprocess import Popen, TimeoutExpired, PIPE
from typing import Tuple, List

import numpy as np
import requests
from tqdm.auto import tqdm
import tiktoken
=======
import logging
import os
import tarfile
from multiprocessing import Pool
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired
from tempfile import NamedTemporaryFile
from typing import List, Optional, Tuple

import numpy as np
import requests
import tiktoken
from tqdm.auto import tqdm
>>>>>>> otherfork/main


def convert_to_markdown(args: Tuple[Path, Path]):
    texfile, mdroot = args
<<<<<<< HEAD
    mdfile = mdroot/f"{texfile.name}.md"
    with Popen(["pandoc", "--wrap=none", "--from", "latex", texfile,
                "--output", mdfile], stderr=PIPE) as proc:
=======
    mdfile = mdroot / f"{texfile.name}.md"
    with Popen(
        ["pandoc", "--wrap=none", "--from", "latex", texfile, "--output", mdfile],
        stderr=PIPE,
    ) as proc:
>>>>>>> otherfork/main
        try:
            proc.communicate(timeout=1)
        except TimeoutExpired:
            proc.kill()


<<<<<<< HEAD

def fetch_arxiv(root: Path, year: int):
    # download latex
    url = f"https://www.cs.cornell.edu/projects/kddcup/download/hep-th-{year}.tar.gz"
    texroot = root/"tex"
=======
def fetch_arxiv(root: Path, year: int):
    # download latex
    url = f"https://www.cs.cornell.edu/projects/kddcup/download/hep-th-{year}.tar.gz"
    texroot = root / "tex"
>>>>>>> otherfork/main
    print("Downloading Arxiv year", year)
    req = requests.get(url, timeout=60)
    with NamedTemporaryFile(suffix=".tar.gz") as f:
        f.write(req.content)
        logging.debug("Tar saved in tempfile %s" % f.name)
        with tarfile.open(f.name) as tar:
            logging.debug("Extracting tarfile")
            tar.extractall(texroot)

    # convert to markdown
<<<<<<< HEAD
    mdroot = root/"md"/str(year)
    mdroot.mkdir(parents=True)
    files = list((texroot/str(year)).iterdir())
    with Pool(os.cpu_count()) as p:
        args = [(texfile, mdroot) for texfile in files]
        for _ in tqdm(p.imap_unordered(convert_to_markdown, args),
                      desc="Converting to markdown", total=len(files)):
=======
    mdroot = root / "md" / str(year)
    mdroot.mkdir(parents=True)
    files = list((texroot / str(year)).iterdir())
    with Pool(os.cpu_count()) as p:
        args = [(texfile, mdroot) for texfile in files]
        for _ in tqdm(
            p.imap_unordered(convert_to_markdown, args),
            desc="Converting to markdown",
            total=len(files),
        ):
>>>>>>> otherfork/main
            pass


def tokenize_arxiv(root: Path, year: int):
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = []
    tokens_val = []
    tokens_test = []
<<<<<<< HEAD
    mds = root/"md"/str(year)
=======
    mds = root / "md" / str(year)
>>>>>>> otherfork/main

    # tokenize
    desc = f"Tokenizing {year}"
    for i, mdpath in enumerate(tqdm(list(mds.iterdir()), desc=desc)):
        with open(mdpath, encoding="utf8") as f:
            text = "".join(f.readlines())
        if i % 10 <= 6:  # train split
            tokens += tokenizer.encode(text)
        elif i % 10 <= 8:  # val split
            tokens_val += tokenizer.encode(text)
        else:  # test split
            tokens_test += tokenizer.encode(text)

    # save to dir
<<<<<<< HEAD
    tpath = root/str(year)
    tpath.mkdir(parents=True)
    for x, name in zip([tokens, tokens_val, tokens_test],
                       ["train", "val", "test"]):
        mem = np.memmap(tpath/f"{name}.npy", dtype=np.uint16, mode="w+",
                        shape=len(x))
=======
    tpath = root / str(year)
    tpath.mkdir(parents=True)
    for x, name in zip([tokens, tokens_val, tokens_test], ["train", "val", "test"]):
        mem = np.memmap(tpath / f"{name}.npy", dtype=np.uint16, mode="w+", shape=len(x))
>>>>>>> otherfork/main
        for i, v in enumerate(x):
            mem[i] = v


def load_arxiv(cachedir: Path, years: Optional[List[int]] = None):
<<<<<<< HEAD
    all_years = list(range(1992, 2004))
    if years is None:
        years = all_years
    assert set(years) <= set(all_years)
    root = cachedir/"arxiv"
=======
    all_years = list(range(1993, 2004))  # 1992 seems to give some problem
    if years is None:
        years = all_years
    assert set(years) <= set(all_years)
    root = cachedir / "arxiv"
>>>>>>> otherfork/main
    root.mkdir(exist_ok=True, parents=True)

    # download all years requested that are not present
    for year in years:
<<<<<<< HEAD
        if not (root/"md"/str(year)).exists():
=======
        if not (root / "md" / str(year)).exists():
>>>>>>> otherfork/main
            fetch_arxiv(root, year)

    # tokenize all years not previously tokenized
    for year in years:
<<<<<<< HEAD
        if not (root/str(year)).exists():
=======
        if not (root / str(year)).exists():
>>>>>>> otherfork/main
            tokenize_arxiv(root, year)

    # load meta
    ret = {}
    for split in ["train", "val"]:
<<<<<<< HEAD
        paths = [root/str(year)/f"{split}.npy" for year in years]
=======
        paths = [root / str(year) / f"{split}.npy" for year in years]
>>>>>>> otherfork/main
        x = [np.memmap(path, dtype=np.uint16, mode="r") for path in paths]
        ret[split] = np.concatenate(x)
    return ret


<<<<<<< HEAD
def get_arxiv_2000():
    return load_arxiv(Path(os.path.dirname(__file__))/"datasets", [2000])


def get_arxiv_full():
    return load_arxiv(Path(os.path.dirname(__file__))/"datasets")
=======
def get_arxiv_2000(datasets_base_dir):
    return load_arxiv(Path(datasets_base_dir), [2000])


def get_arxiv_full(datasets_base_dir):
    return load_arxiv(Path(datasets_base_dir))
>>>>>>> otherfork/main
