import regex as re
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def count_pairs(
    pretoken_counts: dict[tuple[bytes, ...], int]
) -> dict[tuple[bytes, bytes], int]:
    """统计所有相邻 Token 对的加权出现频率。"""

    pair_counts = {}

    for tokens, freq in pretoken_counts.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq

    return pair_counts

def merge_pair(
    tokens: tuple[bytes, ...],
    pair: tuple[bytes, bytes]
) -> tuple[bytes, ...]:
    result = []
    i = 0

    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            result.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return tuple(result)

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    special_tokens = list(dict.fromkeys(special_tokens))

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    if vocab_size < len(vocab):
        raise ValueError("vocab_size is smaller than initial vocabulary")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if special_tokens:
        pattern = "|".join(
            re.escape(token)
            for token in sorted(special_tokens, key=len, reverse=True)
        )
        segments = re.split(pattern, text)
    else:
        segments = [text]

    pretoken_counts = Counter()

    for segment in segments:
        for match in re.finditer(PAT, segment):
            encoded = match.group().encode("utf-8")
            tokens = tuple(bytes([b]) for b in encoded)
            pretoken_counts[tokens] += 1

    while len(vocab) < vocab_size:
        pair_counts = count_pairs(pretoken_counts)

        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))

        merges.append(best_pair)

        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        new_pretoken_counts = Counter()

        for tokens, freq in pretoken_counts.items():
            if best_pair[0] not in tokens:
                new_pretoken_counts[tokens] += freq
                continue

            new_pretoken_counts[merge_pair(tokens, best_pair)] += freq

        pretoken_counts = new_pretoken_counts

    return vocab, merges