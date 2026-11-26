"""Byte-level byte-pair encoding (BPE), the tokenizer used by GPT-2, Llama and most LLMs.

How it works:

  * Text is first turned into UTF-8 bytes, so any string can be represented
    and there are never unknown characters. The starting vocabulary is the
    256 possible byte values.
  * Training repeatedly finds the most frequent pair of adjacent tokens in the
    corpus and merges it into a new token. "t" + "h" -> "th", then
    "th" + "e" -> "the", and so on, until the vocabulary is the size we want.
  * Encoding new text replays those merges in the order they were learned.

Before counting pairs, text is split into chunks (words, numbers, punctuation,
spaces) with a regular expression, and merges never cross chunk boundaries.
That stops tokens like "dog." or "the cat" from forming.
"""

import re
from collections import Counter, defaultdict

from slm.tokenizer import EOS, split_on_eos

# Words with an optional leading space, numbers, punctuation runs, and whitespace.
# Together the alternatives match every character, so no text is ever dropped.
CHUNK_PATTERN = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?(?:[^\w\s]|_)+|\s+(?!\S)|\s+"
)


def split_into_chunks(text):
    return CHUNK_PATTERN.findall(text)


def merge_pair(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    merged, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


def train_bpe(text, vocab_size, verbose=False):
    """Learn BPE merges from `text`. Returns the merges as an ordered list of pairs.

    The final vocabulary is 256 bytes + len(merges) merged tokens + 1 EOS token.
    """
    num_merges = vocab_size - 256 - 1
    assert num_merges >= 0, "vocab_size must be at least 257"

    # Count each distinct chunk once; the work then scales with distinct chunks, not text length.
    chunk_counts = Counter()
    for part in split_on_eos(text):
        if part != EOS:
            chunk_counts.update(split_into_chunks(part))
    words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
    counts = list(chunk_counts.values())

    # How often each adjacent pair occurs, and which words contain it.
    pair_counts = Counter()
    words_with_pair = defaultdict(set)
    for word_index, (word, count) in enumerate(zip(words, counts)):
        for pair in zip(word, word[1:]):
            pair_counts[pair] += count
            words_with_pair[pair].add(word_index)

    merges = []
    for step in range(num_merges):
        if not pair_counts:
            break  # every chunk is already a single token
        best_pair = max(pair_counts, key=pair_counts.get)
        new_id = 256 + step
        merges.append(best_pair)

        # Only words containing the best pair change; update just those.
        for word_index in words_with_pair.pop(best_pair):
            word, count = words[word_index], counts[word_index]
            for pair in zip(word, word[1:]):
                pair_counts[pair] -= count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
            word = merge_pair(word, best_pair, new_id)
            for pair in zip(word, word[1:]):
                pair_counts[pair] += count
                words_with_pair[pair].add(word_index)
            words[word_index] = word

        if verbose and (step + 1) % 500 == 0:
            print(f"  learned {step + 1} / {num_merges} merges")
    return merges


class BPETokenizer:
    def __init__(self, merges):
        self.merges = [tuple(pair) for pair in merges]
        # Rank of each merge: earlier merges are applied first when encoding.
        self.merge_rank = {pair: 256 + i for i, pair in enumerate(self.merges)}

        # The bytes each token stands for.
        self.token_bytes = [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self.token_bytes.append(self.token_bytes[a] + self.token_bytes[b])

        self.eos_id = len(self.token_bytes)
        self.vocab_size = self.eos_id + 1
        self._chunk_cache = {}

    @classmethod
    def train(cls, text, vocab_size, verbose=False):
        return cls(train_bpe(text, vocab_size, verbose))

    def _encode_chunk(self, chunk):
        if chunk in self._chunk_cache:
            return self._chunk_cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned merge that occurs in this chunk.
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.merge_rank.get(p, float("inf")))
            if pair not in self.merge_rank:
                break
            ids = merge_pair(ids, pair, self.merge_rank[pair])
        self._chunk_cache[chunk] = ids
        return ids

    def encode(self, text):
        ids = []
        for part in split_on_eos(text):
            if part == EOS:
                ids.append(self.eos_id)
            else:
                for chunk in split_into_chunks(part):
                    ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids):
        pieces, buffer = [], b""
        for i in ids:
            if i == self.eos_id:
                pieces.append(buffer.decode("utf-8", errors="replace"))
                pieces.append(EOS)
                buffer = b""
            else:
                buffer += self.token_bytes[i]
        pieces.append(buffer.decode("utf-8", errors="replace"))
        return "".join(pieces)

    def to_dict(self):
        return {"type": "bpe", "merges": [list(pair) for pair in self.merges]}

    @classmethod
    def from_dict(cls, d):
        return cls(d["merges"])
