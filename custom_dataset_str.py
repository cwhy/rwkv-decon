import os
from pathlib import Path


def load(base_path: Path, dataset: str = "english"):
    path = base_path / dataset
    books = [f for f in os.listdir(path) if f.endswith('.txt')]

    def read_book_file(filename: str):
        print(f"reading {filename}...")
        encodings = ['utf-8', 'gb2312', 'gbk', 'big5', 'utf-16', 'gb18030']
        for encoding in encodings:
            try:
                with open(path / filename, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                print(f"Failed to read {filename} with encoding: {encoding}")
                continue
        raise Exception(f"Failed to read {filename} with all available encodings")

    text = "\n\n".join(f"{book_name}\n\n {read_book_file(book_name)}" for book_name in books)
    return text
