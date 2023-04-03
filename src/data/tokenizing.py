from string import ascii_letters, digits, punctuation

import tiktoken


_char_decode = dict(enumerate(sorted(set(ascii_letters + digits + punctuation + " \n"))))
_char_encode = {char: i for i, char in _char_decode.items()}


def encode(txt: str, tokenizer: str) -> list[int]:
    """Tokenize `txt` using `tokenizer` {bpe, character} method"""
    if tokenizer == "bpe":
        tokenizer = tiktoken.get_encoding("gpt2")
        return tokenizer.encode(txt)
    if tokenizer == "character":
        return [_char_encode[char] for char in txt if char in _char_encode]
    raise KeyError(f"Tokenizer {tokenizer} unknown")


def decode(x: list[int], tokenizer: str) -> str:
    """Tokenize `txt` using `tokenizer` {bpe, character} method"""
    if tokenizer == "bpe":
        tokenizer = tiktoken.get_encoding("gpt2")
        return tokenizer.decode(x)
    if tokenizer == "character":
        return "".join(_char_decode[i] for i in x if i in _char_decode)
    raise KeyError(f"Tokenizer {tokenizer} unknown")
