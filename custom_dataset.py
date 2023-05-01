import os
from collections import Counter
from pathlib import Path
from typing import NamedTuple, Callable

import jax.numpy as xp

# dataset = "play"
base_path = Path("/Data/nlp/")


def load(dataset: str = "english"):
    path = base_path / dataset
    books = [f for f in os.listdir(path) if f.endswith('.txt')]

    def read_book_file(filename: str):
        print(f"reading {filename}...")
        try:
            with open(path / filename, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(path / filename, 'r', encoding='gb2312') as f:
                    return f.read()
            except UnicodeDecodeError:
                try:
                    with open(path / filename, 'r', encoding='gbk') as f:
                        return f.read()
                except UnicodeDecodeError:
                    try:
                        with open(path / filename, 'r', encoding='big5') as f:
                            return f.read()
                    except UnicodeDecodeError:
                        try:
                            with open(path / filename, 'r', encoding='utf-16') as f:
                                return f.read()
                        except UnicodeDecodeError:
                            try:
                                with open(path / filename, 'r', encoding='gb18030') as f:
                                    return f.read()
                            except UnicodeDecodeError:
                                raise Exception(f"Failed to read {filename} with many encodings")

    text = "\n\n".join(f"{book_name}\n\n {read_book_file(book_name)}" for book_name in books)
    chars = [ch for ch, c in Counter(text).most_common()]
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    def encode(_text: str):
        return [stoi[c] for c in _text]

    def decode(_encoded: list):
        return "".join(itos[i] for i in _encoded)

    return text, encode, decode, vocab_size


def load_jax_cached(dataset: str = "english"):
    text, encode, decode, vocab_size = load(dataset)
    cache_path = base_path / dataset / 'encoded_jax.npy'
    try:
        with open(cache_path, 'rb') as f:
            encoded_jax = xp.load(f)
    except FileNotFoundError:
        encoded = encode(text)
        encoded_jax = xp.array(encoded, dtype=xp.int16)
        print(encoded_jax.shape, encoded_jax.dtype)
        with open(cache_path, 'wb') as fw:
            xp.save(fw, encoded_jax)
    return encoded_jax, encode, decode, vocab_size


class Tokens(NamedTuple):
    ids: list[int]


class Tokenizer(NamedTuple):
    vocab_size: int
    encode: Callable[[str], Tokens]
    decode: Callable[[list[int]], str]

    def get_vocab_size(self) -> int:
        return self.vocab_size


def load_jax_cached_tokenizer(dataset: str = "english") -> tuple[xp.ndarray, Tokenizer]:
    text, encode, decode, vocab_size = load(dataset)
    cache_path = base_path / dataset / 'encoded_jax.npy'
    try:
        with open(cache_path, 'rb') as f:
            encoded_jax = xp.load(f)
    except FileNotFoundError:
        encoded = encode(text)
        encoded_jax = xp.array(encoded, dtype=xp.int16)
        print(encoded_jax.shape, encoded_jax.dtype)
        with open(cache_path, 'wb') as fw:
            xp.save(fw, encoded_jax)
    return encoded_jax, Tokenizer(vocab_size, lambda x: Tokens(encode(x)), decode)
