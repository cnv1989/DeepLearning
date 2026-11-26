"""Generate test vectors for the testbenches from the bit-exact reference model.

core_vectors.hex: one 16-bit word per operation
    [15:14] op      0 = push, 1 = start, 3 = end of test
    [13]    noise_en (for start)
    [12:8]  token_in (for push)
    [4:0]   expected token_out (for start)
uart_in.hex / uart_expected.hex: bytes sent to / expected from the FPGA UART top.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import charset  # noqa: E402
from ref_model import TinyLM  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (prompt, number of tokens to generate, noise)
SCENARIO = [
    ("the cat ", 40, False),
    ("mary had a ", 40, True),
    ("twinkle", 30, False),
    ("", 60, True),
    ("HUMPTY dumpty sat on a", 30, True),
]
GEN_LEN_UART = 24  # must match GEN_LEN used by tb/tb_uart_top.v


def core_vectors(m):
    words, text = [], []
    for prompt, n, noise in SCENARIO:
        toks = [charset.char2tok(c) for c in prompt]   # raw mapping, no whitespace collapsing
        for t in toks:
            m.push(t)
            words.append((0 << 14) | (t << 8))
        out = [m.start(noise) for _ in range(n)]
        words += [(1 << 14) | (int(noise) << 13) | t for t in out]
        text.append(f"{prompt!r} -> {charset.decode(out)!r}  (noise={'on' if noise else 'off'})")
    words.append(3 << 14)
    return words, text


def uart_vectors(m):
    """Mirror of the FPGA controller in rtl/tinylm_uart_top.v:
    printable char -> push + echo the mapped char; '~' -> toggle noise (echo '~');
    CR or LF -> send CR LF, generate GEN_LEN chars, send CR LF."""
    sent = b"the cat\r~mary\r"
    noise, exp = False, bytearray()
    for b in sent:
        if b in (0x0D, 0x0A):
            exp += b"\r\n"
            for _ in range(GEN_LEN_UART):
                exp += charset.tok2char(m.start(noise)).encode()
            exp += b"\r\n"
        elif b == ord("~"):
            noise = not noise
            exp += b"~"
        else:
            t = charset.char2tok(chr(b))
            m.push(t)
            exp += charset.tok2char(t).encode()
    return list(sent), list(exp)


def write_hex(path, values, width):
    with open(path, "w") as f:
        for v in values:
            f.write(f"{v:0{width}x}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "model", "weights", "tt.json"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "tb", "vectors"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    m = TinyLM.load(args.weights)
    words, text = core_vectors(m)
    write_hex(os.path.join(args.outdir, "core_vectors.hex"), words, 4)
    with open(os.path.join(args.outdir, "core_counts.vh"), "w") as f:
        f.write(f"localparam integer CORE_LEN = {len(words)};\n")
    print(f"core_vectors.hex: {len(words)} operations")
    for t in text:
        print("  " + t)

    m.reset()
    sent, exp = uart_vectors(m)
    write_hex(os.path.join(args.outdir, "uart_in.hex"), sent + [0], 2)
    write_hex(os.path.join(args.outdir, "uart_expected.hex"), exp, 2)
    with open(os.path.join(args.outdir, "uart_counts.vh"), "w") as f:
        f.write(f"localparam integer UART_IN_LEN = {len(sent)};\n"
                f"localparam integer UART_EXP_LEN = {len(exp)};\n")
    print(f"uart vectors: send {bytes(sent)!r}, expect {len(exp)} bytes:")
    print("  " + bytes(exp).decode().replace("\r\n", "\\r\\n"))


if __name__ == "__main__":
    main()
