import pickle
import numpy as np
from itertools import islice
from pathlib import Path

# 改成你实际的 Tokenizer 类所在位置
from cs336_basics.Tokenizer.tokenizer import Tokenizer

data_dir = Path("data")

with open(data_dir / "tinystories_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open(data_dir / "tinystories_merges.pkl", "rb") as f:
    merges = pickle.load(f)

tokenizer = Tokenizer(
    vocab=vocab,
    merges=merges,
    special_tokens=["<|endoftext|>"],
)


def tokenize_file(input_path: Path, output_path: Path):
    with open(input_path, "r", encoding="utf-8") as src, \
         open(output_path, "wb") as dst:

        ids = tokenizer.encode_iterable(src)

        while True:
            chunk = np.fromiter(
                islice(ids, 1_000_000),
                dtype=np.uint16,
            )

            if chunk.size == 0:
                break

            chunk.tofile(dst)

    print(f"Saved: {output_path}")


tokenize_file(
    data_dir / "TinyStoriesV2-GPT4-train.txt",
    data_dir / "tinystories_train.bin",
)

tokenize_file(
    data_dir / "TinyStoriesV2-GPT4-valid.txt",
    data_dir / "tinystories_valid.bin",
)