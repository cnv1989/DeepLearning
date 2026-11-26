"""Tokenizers turn text into a list of integer ids, and ids back into text.

CharTokenizer      one token per character. Tiny vocabulary, easy to understand.
BPETokenizer       byte-level byte-pair encoding, trained on your own data
                   (see slm/bpe.py). The standard choice for real models.
TiktokenTokenizer  OpenAI's pretrained GPT-2 BPE (50,257 tokens) via tiktoken.

Every tokenizer has an end-of-text token, EOS = "<|endoftext|>". It marks the
boundary between documents during pretraining, and the end of an answer in
fine-tuning, so the model learns when to stop.
"""

EOS = "<|endoftext|>"


def split_on_eos(text):
    """Split text around EOS markers: "a<|endoftext|>b" -> ["a", EOS, "b"]."""
    parts = []
    for i, piece in enumerate(text.split(EOS)):
        if i > 0:
            parts.append(EOS)
        if piece:
            parts.append(piece)
    return parts


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.char_to_id = {c: i for i, c in enumerate(self.chars)}
        self.eos_id = len(self.chars)  # EOS comes right after the characters
        self.vocab_size = len(self.chars) + 1

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text.replace(EOS, ""))))

    def encode(self, text):
        ids = []
        for part in split_on_eos(text):
            if part == EOS:
                ids.append(self.eos_id)
            else:
                # Characters never seen in training are skipped.
                ids.extend(self.char_to_id[c] for c in part if c in self.char_to_id)
        return ids

    def decode(self, ids):
        return "".join(EOS if i == self.eos_id else self.chars[i] for i in ids)

    def to_dict(self):
        return {"type": "char", "chars": self.chars}


class TiktokenTokenizer:
    def __init__(self, name="gpt2"):
        import tiktoken

        self.name = name
        self.encoding = tiktoken.get_encoding(name)
        self.eos_id = self.encoding.eot_token
        self.vocab_size = self.encoding.n_vocab

    def encode(self, text):
        return self.encoding.encode(text, allowed_special={EOS})

    def decode(self, ids):
        return self.encoding.decode(ids)

    def to_dict(self):
        return {"type": "tiktoken", "name": self.name}


def tokenizer_from_dict(d):
    """Rebuild a tokenizer saved with `to_dict()`."""
    if d["type"] == "char":
        return CharTokenizer(d["chars"])
    if d["type"] == "bpe":
        from slm.bpe import BPETokenizer

        return BPETokenizer.from_dict(d)
    if d["type"] == "tiktoken":
        return TiktokenTokenizer(d["name"])
    raise ValueError(f"unknown tokenizer type: {d['type']}")
