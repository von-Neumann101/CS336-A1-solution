import pickle
from collections.abc import Iterable, Iterator
import regex as re

from cs336_basics.Tokenizer.train_bpe import merge_pair

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer():

    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None
                 ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.bytes_to_id = {
            token: token_id
            for token_id, token in self.vocab.items()
        }
        self.max_token = 0

    @classmethod
    def from_files(
            cls, 
            vocab_filepath: str, 
            merges_filepath: str, 
            special_tokens: list[str] | None = None
            ) -> "Tokenizer":
        """从文件中加载词表和合并规则，创建 Tokenizer 实例"""
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """文本编码为 token ID 列表"""
        special_tokens = list(dict.fromkeys(self.special_tokens))
        
        for token in special_tokens:
            self.vocab[len(self.vocab)] = token.encode("utf-8")

        if special_tokens:
            pattern = "(" + "|".join(
                re.escape(token)
                for token in sorted(self.special_tokens, key=len, reverse=True)
            ) + ")"

            segments = re.split(pattern, text)
        else:
            segments = [text]

        tokens_id = []

        for segment in segments:
            if segment in self.special_tokens:
                tokens_id.append(self.bytes_to_id[segment.encode("utf-8")])
                continue

            for match in re.finditer(PAT, segment):
                encoded = match.group().encode("utf-8")
                tokens = tuple(bytes([b]) for b in encoded)

                for merge in self.merges:
                    tokens = merge_pair(tokens, merge)

                tokens_id.extend(self.bytes_to_id[token] for token in tokens)

        return tokens_id

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pre_pattern = re.compile(PAT)

        specials = sorted(
            set(self.special_tokens or []),
            key=len,
            reverse=True
        )
        specials = [s for s in specials if s]

        special_pattern = (
            re.compile("|".join(re.escape(s) for s in specials))
            if specials else None
        )

        def split_units(text):
            pos = 0

            if special_pattern:
                for match in special_pattern.finditer(text):
                    for m in pre_pattern.finditer(text, pos, match.start()):
                        yield m.end(), m.group(), False

                    yield match.end(), match.group(), True
                    pos = match.end()

            for m in pre_pattern.finditer(text, pos):
                yield m.end(), m.group(), False

        def encode_unit(text, is_special):
            if is_special:
                return [self.bytes_to_id[text.encode("utf-8")]]
            return self.encode(text)

        buffer = ""

        for chunk in iterable:
            buffer += chunk

            if not buffer:
                continue

            pending_start = len(buffer)

            matches = (
                list(special_pattern.finditer(buffer))
                if special_pattern else []
            )

            for special in specials:
                for k in range(1, min(len(special), len(buffer) + 1)):
                    if buffer.endswith(special[:k]):
                        start = len(buffer) - k

                        if not any(
                            m.start() < start < m.end()
                            for m in matches
                        ):
                            pending_start = min(pending_start, start)

            previous = None
            consumed = 0

            for unit in split_units(buffer[:pending_start]):
                if previous is not None:
                    end, text, is_special = previous
                    yield from encode_unit(text, is_special)
                    consumed = end

                previous = unit

            buffer = buffer[consumed:]

        for _, text, is_special in split_units(buffer):
            yield from encode_unit(text, is_special)

    def decode(self, ids: list[int]) -> str:
        return b"".join(
            self.vocab[token_id] for token_id in ids
        ).decode("utf-8", errors="replace")