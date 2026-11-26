"""Character set shared by the Python model and the generated Verilog.

32 symbols -> 5-bit tokens. Token 0 (space) is also the "empty" context value.
"""
CHARS = " abcdefghijklmnopqrstuvwxyz.,'!?"
assert len(CHARS) == 32
V = len(CHARS)
TOK = {c: i for i, c in enumerate(CHARS)}


def char2tok(c: str) -> int:
    """Map any character to a token. Upper case folds to lower case,
    unknown characters become space (token 0). Mirrors rtl/generated/tinylm_charmap.v."""
    c = c.lower()
    return TOK.get(c, 0)


def tok2char(t: int) -> str:
    return CHARS[t]


def encode(text: str) -> list[int]:
    """Text -> tokens, collapsing runs of whitespace/unknowns into one space."""
    out = []
    for c in text:
        t = char2tok(c)
        if t == 0 and out and out[-1] == 0:
            continue
        out.append(t)
    return out


def decode(tokens) -> str:
    return "".join(CHARS[t] for t in tokens)
