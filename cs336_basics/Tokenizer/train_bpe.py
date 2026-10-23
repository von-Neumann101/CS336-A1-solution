import regex as re
from collections import Counter, defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def count_pairs(
    pretoken_counts: dict[tuple[bytes, ...], int]
) -> tuple[
    dict[tuple[bytes, bytes], int],
    dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
]:
    """统计所有相邻 Token 对的加权出现频率以及这个 pair 在的 pretoken。"""
    pair_counts = Counter()
    pair_to_pretokens = defaultdict(set)

    for tokens, freq in pretoken_counts.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
            pair_to_pretokens[pair].add(tokens)

    return pair_counts, pair_to_pretokens

def merge_pair(
    tokens: tuple[bytes, ...],
    pair: tuple[bytes, bytes]
) -> tuple[bytes, ...]:
    """融合 pair 中的两个 token"""
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
    """训练 bpe"""
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

    pair_counts, pair_to_pretokens = count_pairs(pretoken_counts)

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))

        affected = tuple(pair_to_pretokens[best_pair]) # best_pair 在的 pretoken 集合

        merges.append(best_pair)

        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        for token in affected:
            freq = pretoken_counts.pop(token)

            # 原来的相邻对
            old_pairs = list(zip(token, token[1:]))

            for pair in old_pairs:
                pair_counts[pair] -= freq
                if pair_counts[pair] == 0:
                    del pair_counts[pair]

            for pair in set(old_pairs):
                pair_to_pretokens[pair].discard(token)
                if not pair_to_pretokens[pair]:
                    del pair_to_pretokens[pair]

            new_tokens = merge_pair(token, best_pair)
            pretoken_counts[new_tokens] = pretoken_counts.get(new_tokens, 0) + freq

            new_pairs = list(zip(new_tokens, new_tokens[1:]))

            for pair in new_pairs:
                pair_counts[pair] += freq

            for pair in set(new_pairs):
                pair_to_pretokens[pair].add(new_tokens)

    return vocab, merges