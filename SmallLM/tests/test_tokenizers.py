from slm.bpe import BPETokenizer, split_into_chunks
from slm.tokenizer import EOS, CharTokenizer, tokenizer_from_dict

TEXT = "Hello, world! It's 2026.\n\n  The cat sat on the mat_1 — café ☕"


def test_chunking_keeps_every_character():
    assert "".join(split_into_chunks(TEXT)) == TEXT


def test_char_round_trip_with_eos():
    tokenizer = CharTokenizer.from_text(TEXT)
    text = TEXT + EOS + "Hello"
    ids = tokenizer.encode(text)
    assert ids.count(tokenizer.eos_id) == 1
    assert tokenizer.decode(ids) == text


def test_bpe_learns_merges_and_round_trips():
    tokenizer = BPETokenizer.train(TEXT * 20, vocab_size=300)
    # A tiny corpus can run out of pairs to merge before reaching the requested size.
    assert 256 < tokenizer.vocab_size <= 300
    ids = tokenizer.encode(TEXT)
    assert len(ids) < len(TEXT.encode("utf-8"))  # merges made it shorter than raw bytes
    assert tokenizer.decode(ids) == TEXT
    # Unseen text still works, because every byte is in the vocabulary.
    assert tokenizer.decode(tokenizer.encode("zebra 🦓")) == "zebra 🦓"


def test_bpe_eos_and_serialization():
    tokenizer = BPETokenizer.train(TEXT * 5, vocab_size=280)
    text = "the cat" + EOS + "sat"
    ids = tokenizer.encode(text)
    assert tokenizer.eos_id in ids
    restored = tokenizer_from_dict(tokenizer.to_dict())
    assert restored.encode(text) == ids
    assert restored.decode(ids) == text
