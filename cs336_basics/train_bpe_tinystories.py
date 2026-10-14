from train_bpe import train_bpe

import os

data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"
)

# TinyStories
train_path = os.path.join(data_dir, "TinyStoriesV2-GPT4-train.txt")
valid_path = os.path.join(data_dir, "TinyStoriesV2-GPT4-valid.txt")

# OpenWebText
owt_train_path = os.path.join(data_dir, "owt_train.txt")
owt_valid_path = os.path.join(data_dir, "owt_valid.txt")

special_tokens = ["<|endoftext|>"]

ts_vocab, _ = train_bpe(input_path= train_path,vocab_size=10000, special_tokens=special_tokens)

owt_vocab, _ = train_bpe(input_path= owt_train_path,vocab_size=32000, special_tokens=special_tokens)

ts_compression_ratio = len(text.encode("utf-8")) / len(tokenizer.encode(text))
print()