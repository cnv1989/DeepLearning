"""Tokenizers turn text into a list of integer ids, and ids back into text.

CharTokenizer  one token per character. Tiny vocabulary, easy to understand.
BPETokenizer   GPT-2's byte-pair encoding (50,257 tokens) via tiktoken.
               Each token is a common chunk of text, so sequences are shorter.
"""


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.char_to_id = {c: i for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    def encode(self, text):
        # Characters never seen in training are skipped.
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def to_dict(self):
        return {"type": "char", "chars": self.chars}


class BPETokenizer:
    def __init__(self, name="gpt2"):
        import tiktoken

        self.name = name
        self.encoding = tiktoken.get_encoding(name)
        self.vocab_size = self.encoding.n_vocab

    def encode(self, text):
        return self.encoding.encode_ordinary(text)

    def decode(self, ids):
        return self.encoding.decode(ids)

    def to_dict(self):
        return {"type": "bpe", "name": self.name}


def tokenizer_from_dict(d):
    """Rebuild a tokenizer saved with `to_dict()`."""
    if d["type"] == "char":
        return CharTokenizer(d["chars"])
    if d["type"] == "bpe":
        return BPETokenizer(d["name"])
    raise ValueError(f"unknown tokenizer type: {d['type']}")
