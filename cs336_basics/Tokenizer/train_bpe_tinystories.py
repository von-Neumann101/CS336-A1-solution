from cs336_basics.Tokenizer.train_bpe import train_bpe
import time
import tracemalloc
import pickle

tracemalloc.start()
start = time.perf_counter()

vocab, merges = train_bpe(
    "data/TinyStoriesV2-GPT4-train.txt",
    10000,
    ["<|endoftext|>"]
)

elapsed = time.perf_counter() - start
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

with open("data/tinystories_vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

with open("data/tinystories_merges.pkl", "wb") as f:
    pickle.dump(merges, f)

normal_tokens = [
    token for token in vocab.values()
    if token != b"<|endoftext|>"
]

longest_token = max(normal_tokens, key=len)

print(f"训练时间：{elapsed:.2f} 秒")
print(f"Python 内存分配峰值：{peak / 1024**2:.2f} MiB")
print(f"词表大小：{len(vocab)}")
print(f"合并次数：{len(merges)}")
print(f"最长 Token：{longest_token!r}")
print(f"最长 Token 字节数：{len(longest_token)}")
print(f"解码结果：{longest_token.decode('utf-8', errors='replace')}")