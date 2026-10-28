import pickle
from collections.abc import Iterable, Iterator

import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # 不修改外部传入的词表
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(dict.fromkeys(special_tokens or []))

        if "" in self.special_tokens:
            raise ValueError("Special tokens cannot be empty")

        self.bytes_to_id = {
            token: idx
            for idx, token in self.vocab.items()
        }

        # 建立 BPE 合并规则的优先级索引
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

        self._pre_pattern = re.compile(PAT)

        # 注册特殊 Token；已存在的直接复用
        self.special_to_id: dict[str, int] = {}
        next_id = max(self.vocab, default=-1) + 1

        for special in self.special_tokens:
            data = special.encode("utf-8")

            if data not in self.bytes_to_id:
                self.vocab[next_id] = data
                self.bytes_to_id[data] = next_id
                next_id += 1

            self.special_to_id[special] = self.bytes_to_id[data]

        ordered_specials = sorted(
            self.special_tokens,
            key=len,
            reverse=True
        )

        self._special_pattern = (
            re.compile(
                "|".join(re.escape(s) for s in ordered_specials)
            )
            if ordered_specials else None
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def _iter_units(
        self,
        text: str
    ) -> Iterator[tuple[int, str, bool]]:
        """依次生成 (结束位置, 文本片段, 是否为特殊 Token)。"""
        start = 0

        if self._special_pattern is not None:
            for match in self._special_pattern.finditer(text):
                for word in self._pre_pattern.finditer(
                    text, start, match.start()
                ):
                    yield word.end(), word.group(), False

                yield match.end(), match.group(), True
                start = match.end()

        for word in self._pre_pattern.finditer(text, start):
            yield word.end(), word.group(), False

    def _encode_pretoken(self, text: str) -> list[int]:
        """对一个预分词片段执行 BPE。"""
        tokens = [
            bytes((byte,))
            for byte in text.encode("utf-8")
        ]

        while len(tokens) > 1:
            best_pair = None
            best_rank = len(self.merges)

            # 查找当前存在的最高优先级合并对
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair)

                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            # 从左向右合并所有不重叠的 best_pair
            merged = []
            i = 0

            while i < len(tokens):
                if (
                    i + 1 < len(tokens)
                    and (tokens[i], tokens[i + 1]) == best_pair
                ):
                    merged.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1

            tokens = merged

        return [self.bytes_to_id[token] for token in tokens]

    def _encode_unit(
        self,
        text: str,
        is_special: bool
    ) -> list[int]:
        if is_special:
            return [self.special_to_id[text]]

        return self._encode_pretoken(text)

    def encode(self, text: str) -> list[int]:
        """将文本编码成 Token ID 序列。"""
        result = []

        for _, unit, is_special in self._iter_units(text):
            result.extend(self._encode_unit(unit, is_special))

        return result

    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterator[int]:
        """流式编码，正确处理跨 chunk 的文本与特殊 Token。"""
        buffer = ""

        for chunk in iterable:
            buffer += chunk

            if not buffer:
                continue

            # 保留可能跨越 chunk 的特殊 Token 前缀
            pending_start = len(buffer)

            if self._special_pattern is not None:
                matches = list(
                    self._special_pattern.finditer(buffer)
                )

                for special in self.special_tokens:
                    for length in range(
                        1, min(len(special), len(buffer) + 1)
                    ):
                        if buffer.endswith(special[:length]):
                            start = len(buffer) - length

                            if not any(
                                match.start() < start < match.end()
                                for match in matches
                            ):
                                pending_start = min(
                                    pending_start, start
                                )

            # 保留最后一个预分词片段：
            # 单词、缩写或空白可能延续到下一个 chunk
            previous = None
            consumed = 0

            for unit in self._iter_units(
                buffer[:pending_start]
            ):
                if previous is not None:
                    end, text, is_special = previous
                    yield from self._encode_unit(
                        text, is_special
                    )
                    consumed = end

                previous = unit

            buffer = buffer[consumed:]

        # 处理最终剩余内容
        for _, text, is_special in self._iter_units(buffer):
            yield from self._encode_unit(text, is_special)

    def decode(self, ids: list[int]) -> str:
        """将 Token ID 序列解码为字符串。"""
        return b"".join(
            self.vocab[idx] for idx in ids
        ).decode("utf-8", errors="replace")